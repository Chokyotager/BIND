import numpy as np

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

import dataset as dataset
from model_graph import ConditionModel

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

from torch_geometric.data import Batch

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

esm_device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

esm_tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

esm_model.to(esm_device)
esm_model.eval()

iterations = 100000
learning_rate = 1e-4
batch_size = 1
grad_accumulation_steps = 256

class MaskLoss (torch.nn.Module):

    def __init__ (self, loss):
        super().__init__()

        self.loss = loss

    def forward (self, yhat, y):

        y = torch.Tensor(y).to(device)
        mask = (y > -999).float().to(device)

        loss = self.loss(yhat.float(), y) * mask

        loss = torch.sum(loss)
        num_non_zero = torch.sum(mask)

        return loss / num_non_zero if num_non_zero > 0 else torch.tensor(0.0)

model = ConditionModel()
loss_classifier = nn.BCEWithLogitsLoss()

loss = MaskLoss(nn.HuberLoss(delta=2.0, reduction="none"))
mse_loss = MaskLoss(nn.MSELoss(reduction="none"))

loss.to(device)
mse_loss.to(device)

esm_model = esm_model.to(esm_device)
model.to(device)
loss_classifier.to(device)

optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=iterations)

scaler = torch.cuda.amp.GradScaler()

def forward_one_step (sequences, graphs):

    encoded_input = esm_tokeniser(sequences, padding="longest", return_tensors="pt")
    esm_output = esm_model.forward(**encoded_input.to(esm_device), output_hidden_states=True)
    hidden_states = esm_output.hidden_states

    hidden_states = [x.detach().to(device) for x in hidden_states]

    attention_mask = encoded_input["attention_mask"].to(device)
    return model.forward(graphs, hidden_states, attention_mask)

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):

    for i in range(iterations):

        try:

            all_loss = list()
            all_rmse = list()

            all_y_classifications = list()
            all_yhat_classifications = list()

            for j in range(grad_accumulation_steps):

                current_batch = dataset.get_train_batch(amount=batch_size)

                sequences = [x[0] for x in current_batch]
                graphs = Batch.from_data_list([x[1] for x in current_batch]).detach().to(device)

                y_ki = torch.from_numpy(np.asarray([x[2] for x in current_batch])).to(torch.float).unsqueeze(1).to(device)
                y_ic50 = torch.from_numpy(np.asarray([x[3] for x in current_batch])).to(torch.float).unsqueeze(1).to(device)
                y_kd = torch.from_numpy(np.asarray([x[4] for x in current_batch])).to(torch.float).unsqueeze(1).to(device)
                y_ec50 = torch.from_numpy(np.asarray([x[5] for x in current_batch])).to(torch.float).unsqueeze(1).to(device)
                y_classification = torch.from_numpy(np.asarray([x[6] for x in current_batch])).to(torch.float).unsqueeze(1).to(device)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    yhat_ki, yhat_ic50, yhat_kd, yhat_ec50, yhat_classification = forward_one_step(sequences, graphs)

                yhat_classification = torch.clamp(yhat_classification, min=-100, max=100)

                ki_loss = loss(yhat_ki, y_ki)
                ic50_loss = loss(yhat_ic50, y_ic50)
                kd_loss = loss(yhat_kd, y_kd)
                ec50_loss = loss(yhat_ec50, y_ec50)
                classification_loss = loss_classifier(yhat_classification, y_classification)

                current_loss = ki_loss + ic50_loss + kd_loss + ec50_loss + classification_loss

                ki_rmse = torch.sqrt(mse_loss(yhat_ki, y_ki))
                ic50_rmse = torch.sqrt(mse_loss(yhat_ic50, y_ic50))
                kd_rmse = torch.sqrt(mse_loss(yhat_kd, y_kd))
                ec50_rmse = torch.sqrt(mse_loss(yhat_ec50, y_ec50))

                current_rmse = ki_rmse + ic50_rmse + kd_rmse + ec50_rmse

                yhat_bin_classification = yhat_classification.sigmoid().detach().cpu().numpy()
                
                all_y_classifications.append(y_classification.cpu().numpy())
                all_yhat_classifications.append(yhat_bin_classification)

                all_loss.append(current_loss.detach())
                all_rmse.append(current_rmse.detach())

                scaler.scale(current_loss / grad_accumulation_steps).backward()

            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad()

            number_positives = int(np.sum(np.asarray(all_y_classifications)))

            print(i, sum(all_loss) / len(all_loss))

            ap = average_precision_score(np.asarray(all_y_classifications).flatten(), np.asarray(all_yhat_classifications).flatten())
            auroc = roc_auc_score(np.asarray(all_y_classifications).flatten(), np.asarray(all_yhat_classifications).flatten())
            accuracy = accuracy_score(np.asarray(all_y_classifications).flatten(), np.round(np.asarray(all_yhat_classifications).flatten()))

            writer.add_scalar("Loss/train", sum(all_loss) / len(all_loss), i)
            writer.add_scalar("RMSE/train", sum(all_rmse) / number_positives, i)

            writer.add_scalar("AUROC/train", auroc, i)
            writer.add_scalar("AP/train", ap, i)
            writer.add_scalar("Accuracy/train", accuracy, i)

            writer.add_scalar("Hyperparams/learning_rate", scheduler.get_last_lr()[0], i)
            writer.add_scalar("Hyperparams/temperature", model.temperature, i)

            scheduler.step()

            if i % 10 == 0:

                all_loss = list()
                all_rmse = list()

                all_y_classifications = list()
                all_yhat_classifications = list()

                for j in range(grad_accumulation_steps):

                    current_batch = dataset.get_validation_batch(amount=batch_size)

                    sequences = [x[0] for x in current_batch]
                    graphs = Batch.from_data_list([x[1] for x in current_batch]).detach().to(device)

                    y_ki = torch.from_numpy(np.asarray([x[2] for x in current_batch])).to(torch.float).unsqueeze(1).to(device)
                    y_ic50 = torch.from_numpy(np.asarray([x[3] for x in current_batch])).to(torch.float).unsqueeze(1).to(device)
                    y_kd = torch.from_numpy(np.asarray([x[4] for x in current_batch])).to(torch.float).unsqueeze(1).to(device)
                    y_ec50 = torch.from_numpy(np.asarray([x[5] for x in current_batch])).to(torch.float).unsqueeze(1).to(device)
                    y_classification = torch.from_numpy(np.asarray([x[6] for x in current_batch])).to(torch.float).unsqueeze(1).to(device)

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        yhat_ki, yhat_ic50, yhat_kd, yhat_ec50, yhat_classification = forward_one_step(sequences, graphs)

                    ki_loss = loss(yhat_ki, y_ki)
                    ic50_loss = loss(yhat_ic50, y_ic50)
                    kd_loss = loss(yhat_kd, y_kd)
                    ec50_loss = loss(yhat_ec50, y_ec50)
                    classification_loss = loss_classifier(yhat_classification, y_classification)

                    yhat_classification = torch.clamp(yhat_classification, min=-100, max=100)

                    classification_loss = loss_classifier(yhat_classification, y_classification)

                    current_loss = ki_loss + ic50_loss + kd_loss + ec50_loss + classification_loss

                    ki_rmse = torch.sqrt(mse_loss(yhat_ki, y_ki))
                    ic50_rmse = torch.sqrt(mse_loss(yhat_ic50, y_ic50))
                    kd_rmse = torch.sqrt(mse_loss(yhat_kd, y_kd))
                    ec50_rmse = torch.sqrt(mse_loss(yhat_ec50, y_ec50))

                    current_rmse = ki_rmse + ic50_rmse + kd_rmse + ec50_rmse

                    yhat_bin_classification = yhat_classification.sigmoid().detach().cpu().numpy()
                
                    all_y_classifications.append(y_classification.cpu().numpy())
                    all_yhat_classifications.append(yhat_bin_classification)

                    all_loss.append(current_loss.detach())
                    all_rmse.append(current_rmse.detach())

                number_positives = int(np.sum(np.asarray(all_y_classifications)))

                ap = average_precision_score(np.asarray(all_y_classifications).flatten(), np.asarray(all_yhat_classifications).flatten())
                auroc = roc_auc_score(np.asarray(all_y_classifications).flatten(), np.asarray(all_yhat_classifications).flatten())
                accuracy = accuracy_score(np.asarray(all_y_classifications).flatten(), np.round(np.asarray(all_yhat_classifications).flatten()))

                writer.add_scalar("Loss/validation", sum(all_loss) / len(all_loss), i)
                writer.add_scalar("RMSE/validation", sum(all_rmse) / number_positives, i)

                writer.add_scalar("AUROC/validation", auroc, i)
                writer.add_scalar("AP/validation", ap, i)
                writer.add_scalar("Accuracy/validation", accuracy, i)

        except Exception as error:

            print(error)
            continue
        
        if i % 1000 == 0 and i != 0:
            torch.save(model, f"saves/model_{i}.pth")

    torch.save(model, "saves/model_final.pth")