# Binding INteraction Determination
import os

import numpy as np
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

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

parser = argparse.ArgumentParser(prog="python3 run_bind.py", description="Binding INteraction Determination version beta")
parser.add_argument("--proteins", type=str, help="Input protein FASTA file", required=True)
parser.add_argument("--ligands", type=str, help="Input ligand SMILES file", required=True)
parser.add_argument("--output", type=str, help="Output file", required=True)
parser.add_argument("--truncate", type=int, default=4096, help="Truncate length")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size of number of ligands")
parser.add_argument("--precision", type=int, default=5, help="Output precision")
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

print("")
print("██████╗░██╗███╗░░██╗██████╗░\n██╔══██╗██║████╗░██║██╔══██╗\n██████╦╝██║██╔██╗██║██║░░██║\n██╔══██╗██║██║╚████║██║░░██║\n██████╦╝██║██║░╚███║██████╔╝\n╚═════╝░╚═╝╚═╝░░╚══╝╚═════╝░")
print("(Binding INteraction Determination - Version 1.4)")
print("Manuscript: https://doi.org/10.1101/2024.04.16.589765")
print("")

print("Transformers version:", transformers.__version__)
print("Torch version:", torch.__version__)
print("NumPy version:", np.__version__)
print("Torch Geometric version:", torch_geometric.__version__)
print("NetworkX version:", networkx.__version__)
print("")
print("Total number of proteins:", len(sequences))
print("Total number of ligands:", len(all_smiles))
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

    for j in tqdm(range(0, len(all_smiles), args.batch_size), ascii=" ▖▘▝▗▚▞█", smoothing=0.1):

      current_batch_smiles = [x for x in all_smiles[j:j+args.batch_size]]
      current_batch = [get_graph(x) for x in current_batch_smiles]
      current_batch_size = len(current_batch)

      repeated_hidden_states = [x.repeat(current_batch_size, 1, 1) for x in hidden_states]
      repeated_attention_mask = attention_mask.repeat(current_batch_size, 1)

      current_graphs = Batch.from_data_list(current_batch).to(device).detach()
      output = model.forward(current_graphs, repeated_hidden_states, repeated_attention_mask)

      output = [x.detach().cpu().numpy() for x in output]

      for k in range(current_batch_size):

        smiles = current_batch_smiles[k]
        
        current_output = [x[k][0] for x in output]
        probability = sigmoid(current_output[-1])

        current_output = current_output + [probability]

        all_scores.append([current_id, smiles] + [np.array2string(x, precision=args.precision) for x in current_output])

open(args.output, "w+").write("\n".join("\t".join(x) for x in all_scores))