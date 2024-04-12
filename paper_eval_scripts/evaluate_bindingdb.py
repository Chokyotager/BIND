import json

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

from dataset_test import test_dataset

test_dataset = test_dataset.items()

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

test_dataset = sorted(test_dataset, key=lambda x: x[1]["sequence"])

proteins = list(set([x[1]["sequence"] for x in test_dataset]))
protein_states = dict()

print("Evaluating...")

previous_sequence = None

all_results = list()

for test_datapoint in tqdm(test_dataset):

    smiles = test_datapoint[1]["smiles"]
    graph = get_graph(smiles)

    protein_sequence = test_datapoint[1]["sequence"]
    
    if protein_sequence != previous_sequence:

        encoded_input = esm_tokeniser([protein_sequence], padding="longest", truncation=False, return_tensors="pt")
        esm_output = esm_model.forward(**encoded_input.to(esm_device), output_hidden_states=True)
        hidden_states = esm_output.hidden_states

        hidden_states = [x.to(device).detach() for x in hidden_states]
        attention_mask = encoded_input["attention_mask"].to(device)

    previous_sequence = protein_sequence

    current_graph = Batch.from_data_list([graph]).to(device)
    output = model.forward(current_graph, hidden_states, attention_mask)

    classifier_output = output[4].sigmoid().detach().cpu().numpy()[0][0]
    ki = output[0].detach().cpu().numpy()[0][0]
    ic50 = output[1].detach().cpu().numpy()[0][0]
    kd = output[2].detach().cpu().numpy()[0][0]
    ec50 = output[3].detach().cpu().numpy()[0][0]

    results = {

        "classifier": float(classifier_output),
        "ki": float(ki),
        "ic50": float(ic50),
        "kd": float(kd),
        "ec50": float(ec50)

    }

    test_datapoint[1]["results"] = results
    all_results.append(test_datapoint[1])

open("bindingdb_results.tsv", "w+").write(json.dumps(all_results, indent=4))