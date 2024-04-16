import os

import torch
from transformers import AutoModel, AutoTokenizer

from tqdm import tqdm

from sklearn import metrics

from torch_geometric.data import Batch

from Bio import SeqIO

import matplotlib.pyplot as plt
import json

import logging
logging.getLogger("pysmiles").setLevel(logging.CRITICAL)

import loading
from data import BondType

from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.data import Data

def get_graph (smiles):

    graph = loading.get_data(smiles, apply_paths=False, parse_cis_trans=False, unknown_atom_is_dummy=True)
    x, a, e = loading.convert(*graph, bonds=[BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC, BondType.NOT_CONNECTED])

    x = torch.Tensor(x)
    a = dense_to_sparse(torch.Tensor(a))[0]
    e = torch.Tensor(e)

    # Given an xae
    graph = Data(x=x, edge_index=a, edge_features=e)

    return graph

esm_tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

esm_model.eval()

esm_device = torch.device("cpu")
device = torch.device("cuda:1")

esm_model = esm_model.to(esm_device)
model = torch.load("saves/model_final.pth", map_location=device)
model.to(device)

model.eval()

proteins = os.listdir("DUD-E-fastas")

for protein in sorted(proteins):

    dud_id = protein.replace(".fasta", "")

    print(dud_id)

    sequences = [str(x.seq) for x in list(SeqIO.parse("DUD-E-fastas/" + protein, "fasta"))]

    sequences = sorted(sequences, key=lambda x: len(x), reverse=True)

    sequence = ".".join(sequences)

    encoded_input = esm_tokeniser([sequence], padding="longest", truncation=False, return_tensors="pt")
    esm_output = esm_model.forward(**encoded_input.to(esm_device), output_hidden_states=True)
    hidden_states = esm_output.hidden_states

    hidden_states = [x.to(device).detach() for x in hidden_states]

    attention_mask = encoded_input["attention_mask"].to(device)

    actives = [x.split("\t") for x in open("AD_smiles/" + dud_id + "_actives.sdf.smi").read().split("\n") if len(x.split("\t")) > 1]
    decoys = [x.split("\t") for x in open("AD_smiles/" + dud_id + "_AD.sdf.smi").read().split("\n") if len(x.split("\t")) > 1]

    ground_truth = [0] * len(actives) + [1] * len(decoys)
    predictions = list()
    other_predictions = list()
    
    for active in tqdm(actives):

        smiles = active[0]
        graph = get_graph(smiles)

        current_graph = Batch.from_data_list([graph]).to(device)
        output = model.forward(current_graph, hidden_states, attention_mask)

        classifier_output = output[4].sigmoid().detach().cpu().numpy()[0][0]
        predictions.append(classifier_output)

        ki = output[0].detach().cpu().numpy()[0][0]
        ic50 = output[1].detach().cpu().numpy()[0][0]
        kd = output[2].detach().cpu().numpy()[0][0]
        ec50 = output[3].detach().cpu().numpy()[0][0]

        other_predictions.append([float(ki), float(ic50), float(kd), float(ec50)])

    for decoy in tqdm(decoys):

        smiles = decoy[0]
        graph = get_graph(smiles)

        current_graph = Batch.from_data_list([graph]).to(device)
        output = model.forward(current_graph, hidden_states, attention_mask)

        classifier_output = output[4].sigmoid().detach().cpu().numpy()[0][0]
        predictions.append(classifier_output)

        ki = output[0].detach().cpu().numpy()[0][0]
        ic50 = output[1].detach().cpu().numpy()[0][0]
        kd = output[2].detach().cpu().numpy()[0][0]
        ec50 = output[3].detach().cpu().numpy()[0][0]

        other_predictions.append([float(ki), float(ic50), float(kd), float(ec50)])

    auroc = metrics.roc_auc_score(ground_truth, predictions)
    ap = metrics.average_precision_score(ground_truth, predictions)

    fpr, tpr, _ = metrics.roc_curve(ground_truth, predictions)
    roc_auc = metrics.auc(fpr, tpr)

    mcc = metrics.matthews_corrcoef(ground_truth, [round(x) for x in predictions])

    paired = [(x, y) for x, y in zip(predictions, ground_truth)]
    paired = sorted(paired, key=lambda x: x[0])

    number_actives = len(actives)
    number_decoys = len(decoys)

    base_factor = number_actives / (number_decoys + number_actives)
    
    success_cutoff_ratio = 0.01
    cutoff_amount = round(success_cutoff_ratio * len(paired))

    sr = len([x for x in paired[:cutoff_amount] if x[1] == 0]) / cutoff_amount
    print("Success rate 0.01:", sr)
    print("Enrichment factor 0.01:", sr / base_factor)

    sr_1 = sr
    ef_1 = sr / base_factor

    success_cutoff_ratio = 0.05
    cutoff_amount = round(success_cutoff_ratio * len(paired))

    sr = len([x for x in paired[:cutoff_amount] if x[1] == 0]) / cutoff_amount
    print("Success rate 0.05:", sr)
    print("Enrichment factor 0.05:", sr / base_factor)

    sr_5 = sr
    ef_5 = sr / base_factor

    success_cutoff_ratio = 0.1
    cutoff_amount = round(success_cutoff_ratio * len(paired))

    sr = len([x for x in paired[:cutoff_amount] if x[1] == 0]) / cutoff_amount
    print("Success rate 0.1:", sr)
    print("Enrichment factor 0.1:", sr / base_factor)

    sr_10 = sr
    ef_10 = sr / base_factor

    # Plotting the ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = {:.2f})".format(roc_auc))
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    plt.legend(loc="lower right")

    plt.savefig("aurocs/" + dud_id + ".png", dpi=300)
    plt.close()

    writable = {

        "auroc": auroc,
        "ap": ap,
        "mcc": mcc,
        "sr_1": sr_1,
        "ef_1": ef_1,
        "sr_5": sr_5,
        "ef_5": ef_5,
        "sr_10": sr_10,
        "ef_10": ef_10,

        "ground_truth": [float(x) for x in ground_truth],
        "predictions": [float(x) for x in predictions],
        "all_predictions": other_predictions

    }

    print("AUROC", auroc)
    print("AP", ap)
    print("MCC", mcc)

    open("DUD-AD_results/" + dud_id + ".json", "w+").write(json.dumps(writable, indent=4))



