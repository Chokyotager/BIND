# Binding INteraction Determination
import os

import torch, torch_geometric, transformers, networkx
from transformers import logging, AutoModel, AutoTokenizer

logging.set_verbosity_error()

from torch_geometric.data import Batch
import sys

import logging
logging.getLogger("pysmiles").setLevel(logging.CRITICAL)

import loading
from data import BondType

from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.data import Data

from tqdm import tqdm

from Bio import SeqIO
import argparse

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

parser = argparse.ArgumentParser(prog="python3 run_bind.py", description="Binding INteraction Determination version beta")
parser.add_argument("--proteins", type=str, help="Input protein FASTA file", required=True)
parser.add_argument("--ligands", type=str, help="Input ligand SMILES file", required=True)
parser.add_argument("--output", type=str, help="Output file", required=True)
parser.add_argument("--truncate", type=int, default=4096, help="Truncate length")
parser.add_argument("--esm_device", type=str, default="cpu", help="Torch device to run ESM-2 on, defaults to 'cpu'")
parser.add_argument("--bind_device", type=str, default="cpu", help="Torch device to run BIND on, defaults to 'cpu'")

if len(sys.argv) < 2:
    parser.print_usage()
    sys.exit(1)

args = parser.parse_args()

protein_file = args.proteins
ligand_file = args.ligands

sequences = list(SeqIO.parse(protein_file, "fasta"))
all_smiles = [x.split()[0] for x in open(ligand_file).read().split("\n") if len(x.split()) > 0]

def get_graph (smiles):

    graph = loading.get_data(smiles, apply_paths=False, parse_cis_trans=False, unknown_atom_is_dummy=True)
    x, a, e = loading.convert(*graph, bonds=[BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC, BondType.NOT_CONNECTED])

    x = torch.Tensor(x)
    a = dense_to_sparse(torch.Tensor(a))[0]
    e = torch.Tensor(e)

    # Given an xae
    graph = Data(x=x, edge_index=a, edge_features=e)

    return graph

script_directory = os.path.dirname(os.path.realpath(__file__))

esm_tokeniser = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

esm_model.eval()

esm_device = torch.device(args.esm_device)
device = torch.device(args.bind_device)

esm_model = esm_model.to(esm_device)
model = torch.load(script_directory + "/saves/BIND_checkpoint_12042024.pth", map_location=device)
model.eval()
model.to(device)

print("\n")
print("██████╗░██╗███╗░░██╗██████╗░\n██╔══██╗██║████╗░██║██╔══██╗\n██████╦╝██║██╔██╗██║██║░░██║\n██╔══██╗██║██║╚████║██║░░██║\n██████╦╝██║██║░╚███║██████╔╝\n╚═════╝░╚═╝╚═╝░░╚══╝╚═════╝░")
print("(Version 1.1b)")
print("\n")

print("Transformers version:", transformers.__version__)
print("Torch version:", torch.__version__)
print("Torch Geometric version:", torch_geometric.__version__)
print("NetworkX version:", networkx.__version__)
print("\n")

all_scores = [["Input protein", "Input SMILES", "pKi", "pIC50", "pKd", "pEC50", "Logit", "Non-binder probability"]]

for i in range(len(sequences)):
    
    sequence = sequences[i]

    current_id = sequence.description
    current_sequence = str(sequence.seq)[:args.truncate]

    print(f"[{i + 1} / {len(sequences)}] {current_id}")

    encoded_input = esm_tokeniser([current_sequence], padding="longest", truncation=False, return_tensors="pt")
    esm_output = esm_model.forward(**encoded_input.to(esm_device), output_hidden_states=True)
    hidden_states = esm_output.hidden_states

    hidden_states = [x.to(device).detach() for x in hidden_states]
    attention_mask = encoded_input["attention_mask"].to(device)

    for smiles in tqdm(all_smiles, ascii=" ▖▘▝▗▚▞█"):
      
      smiles_graph = get_graph(smiles)

      current_graph = Batch.from_data_list([smiles_graph]).to(device).detach()
      output = model.forward(current_graph, hidden_states, attention_mask)

      output = [float(x.detach().cpu().numpy()[0][0]) for x in output]
      probability = sigmoid(output[-1])

      all_scores.append([current_id, smiles] + [str(x) for x in output] + [str(probability)])

open(args.output, "w+").write("\n".join("\t".join(x) for x in all_scores))