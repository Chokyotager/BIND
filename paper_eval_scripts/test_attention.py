import torch
from transformers import AutoModel, AutoTokenizer

from torch_geometric.data import Batch

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

sequence = "IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQFISASKSIVHPSYNSNTLNNDIMLIKLKSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAPILSDSSCKSAYPGQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN"
smiles = "	[H]/N=C(\c1ccc(cc1)NC(=O)c2cc(c(cc2c3ccc(nc3C(=O)O)C(=O)NCC4CC4)OC)C=C)/N"

esm_tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

esm_model.eval()

esm_device = torch.device("cuda:0")
device = torch.device("cuda:0")

esm_model = esm_model.to(esm_device)
model = torch.load("saves/model_final.pth", map_location=device)
model.to(device)

model.eval()

encoded_input = esm_tokeniser([sequence], padding="longest", truncation=False, return_tensors="pt")
esm_output = esm_model.forward(**encoded_input.to(esm_device), output_hidden_states=True)
hidden_states = esm_output.hidden_states

hidden_states = [x.to(device).detach() for x in hidden_states]
attention_mask = encoded_input["attention_mask"].to(device)

graph = get_graph(smiles)
current_graph = Batch.from_data_list([graph]).to(device)

attentions = model.forward(current_graph, hidden_states, attention_mask, return_attentions=True)[-1]

stacked_attentions = torch.stack(attentions, dim=1)

# Average across atoms
meaned_attentions = torch.amax(stacked_attentions, dim=2)

# Average across layers
meaned_attentions = torch.mean(meaned_attentions, dim=1)

residue_attentions = meaned_attentions[0, 1:-1]

print(residue_attentions.detach().cpu().numpy().tolist())