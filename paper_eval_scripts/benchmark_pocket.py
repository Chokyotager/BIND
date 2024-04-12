import torch
from transformers import AutoModel, AutoTokenizer

from torch_geometric.data import Batch

import logging
logging.getLogger("pysmiles").setLevel(logging.CRITICAL)

import loading
from data import BondType

from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.data import Data

from Bio import SeqIO
import json

import os

def get_graph (smiles):

    graph = loading.get_data(smiles, apply_paths=False, parse_cis_trans=False, unknown_atom_is_dummy=True)
    x, a, e = loading.convert(*graph, bonds=[BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC, BondType.NOT_CONNECTED])

    x = torch.Tensor(x)
    a = dense_to_sparse(torch.Tensor(a))[0]
    e = torch.Tensor(e)

    # Given an xae
    graph = Data(x=x, edge_index=a, edge_features=e)

    return graph

def indices_of_top_values (lst, x):

    indexed_lst = list(enumerate(lst))  # Enumerate the list to keep track of indices
    sorted_indices = sorted(indexed_lst, key=lambda x: x[1], reverse=True)  # Sort based on values
    top_indices = [index for index, _ in sorted_indices[:x]]  # Get indices of top X elements

    return top_indices

esm_tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

esm_model.eval()

esm_device = torch.device("cpu")
device = torch.device("cuda:0")

esm_model = esm_model.to(esm_device)
model = torch.load("saves/model_final.pth", map_location=device)
model.to(device)

model.eval()

datapoints = os.listdir("pocket_coreset_benchmark")

for datapoint in sorted(datapoints):

    sequence_file = f"pocket_coreset_benchmark/{datapoint}/{datapoint}.fasta"
    sequence = str(list(SeqIO.parse(sequence_file, "fasta"))[0].seq)

    smiles_file = f"pocket_coreset_benchmark/{datapoint}/{datapoint}.smi"
    smiles = open(smiles_file).read().split()[0]

    pocket_residues_file = f"pocket_coreset_benchmark/{datapoint}/{datapoint}_8A.json"
    pocket_residues = json.loads(open(pocket_residues_file).read())

    encoded_input = esm_tokeniser([sequence], padding="longest", truncation=False, return_tensors="pt")
    esm_output = esm_model.forward(**encoded_input.to(esm_device), output_hidden_states=True)
    hidden_states = esm_output.hidden_states

    hidden_states = [x.to(device).detach() for x in hidden_states]
    attention_mask = encoded_input["attention_mask"].to(device)

    graph = get_graph(smiles)
    current_graph = Batch.from_data_list([graph]).to(device)

    classification, attentions = model.forward(current_graph, hidden_states, attention_mask, return_attentions=True)[-2:]

    stacked_attentions = torch.stack(attentions, dim=1)

    # Max across atoms
    meaned_attentions = torch.amax(stacked_attentions, dim=2)

    # Average across layers
    meaned_attentions = torch.amax(meaned_attentions, dim=1)

    residue_attentions = meaned_attentions[0, 1:-1]

    residue_attentions = residue_attentions.detach().cpu().numpy().tolist()

    top_residues = indices_of_top_values(residue_attentions, len(pocket_residues))

    included_residues = [x for x in pocket_residues if x in top_residues]
    hit_rate = len(included_residues) / len(pocket_residues)

    print(datapoint, hit_rate, classification.sigmoid().detach().cpu().numpy()[0][0])