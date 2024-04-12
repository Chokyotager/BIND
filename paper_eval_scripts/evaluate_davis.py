import csv

import torch
from transformers import AutoModel, AutoTokenizer

import logging
logging.getLogger("pysmiles").setLevel(logging.CRITICAL)

import loading
from data import BondType

from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.data import Data

from torch_geometric.data import Batch

from tqdm import tqdm

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
device = torch.device("cpu")

esm_model = esm_model.to(esm_device)
model = torch.load("saves/model_final.pth", map_location=device)
model.to(device)

model.eval()

davis_data = list(csv.reader(open("davis_kiba/davis.csv")))[1:]

proteins = list(set([x[4] for x in davis_data]))
protein_states = dict()

print("Latentising proteins")
for protein in tqdm(proteins):

    encoded_input = esm_tokeniser([protein], padding="longest", truncation=False, return_tensors="pt")
    esm_output = esm_model.forward(**encoded_input.to(esm_device), output_hidden_states=True)
    hidden_states = esm_output.hidden_states

    hidden_states = [x.to(device).detach() for x in hidden_states]
    attention_mask = encoded_input["attention_mask"].to(device)

    protein_states[protein] = {"mask": attention_mask, "states": hidden_states}

print("Evaluating")

writable = [["SMILES", "Protein", "Classifier", "Ki", "IC50", "Kd", "EC50"]]

for davis_datapoint in tqdm(davis_data):

    smiles = davis_datapoint[2]
    graph = get_graph(smiles)

    protein_sequence = davis_datapoint[4]
    current_protein = protein_states[protein_sequence]

    hidden_states = current_protein["states"]
    attention_mask = current_protein["mask"]

    current_graph = Batch.from_data_list([graph]).to(device)
    output = model.forward(current_graph, hidden_states, attention_mask)

    classifier_output = output[4].sigmoid().detach().cpu().numpy()[0][0]
    ki = output[0].detach().cpu().numpy()[0][0]
    ic50 = output[1].detach().cpu().numpy()[0][0]
    kd = output[2].detach().cpu().numpy()[0][0]
    ec50 = output[3].detach().cpu().numpy()[0][0]

    writable.append([smiles, protein_sequence, str(classifier_output), str(ki), str(ic50), str(kd), str(ec50)])

open("davis_results.tsv", "w+").write("\n".join(["\t".join(x) for x in writable]))