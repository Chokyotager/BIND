import json
import random
import math

from tqdm import tqdm

import logging
logging.getLogger("pysmiles").setLevel(logging.CRITICAL)

import loading
from data import BondType

import torch

from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.data import Data

# Optimisation
length_cutoff = 2048

print("Loading main database")

main_database = json.loads(open("reformatted_binding_db.json").read())
all_smiles_pairs = dict()

for key in main_database.keys():

    sequence = main_database[key]["sequence"].upper()

    if sequence not in all_smiles_pairs.keys():
        all_smiles_pairs[sequence] = list()

    all_smiles_pairs[sequence].append(main_database[key]["smiles"])

all_smiles = list(set([x["smiles"] for x in main_database.values()]))

def get_graph (smiles):

    graph = loading.get_data(smiles, apply_paths=False, parse_cis_trans=False, unknown_atom_is_dummy=True)
    x, a, e = loading.convert(*graph, bonds=[BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC, BondType.NOT_CONNECTED])

    x = torch.Tensor(x)
    a = dense_to_sparse(torch.Tensor(a))[0]
    e = torch.Tensor(e)

    # Given an xae
    graph = Data(x=x, edge_index=a, edge_features=e)

    return graph

smiles_graphs = dict()

for allowed_smiles in tqdm(all_smiles):

    try:

        graph = get_graph(allowed_smiles)

        if graph.edge_index.shape[1] == 0:
            continue

        smiles_graphs[allowed_smiles] = graph

    except:

        continue

allowed_smiles_set = set(smiles_graphs.keys())
allowed_smiles_list = list(allowed_smiles_set)

current_set = [x for x in main_database.items() if x[1]["smiles"] in allowed_smiles_set]

for key in main_database.keys():

    main_database[key]["length"] = len(main_database[key]["sequence"])

print("Reformatting main database")

random.Random(0).shuffle(current_set)

train_set = current_set[:math.floor(0.9 * len(current_set))]
validation_set = current_set[math.floor(0.9 * len(current_set)):math.floor(0.92 * len(current_set))]
test_set = current_set[math.floor(0.92 * len(current_set)):]

test_dataset = {x[0]: x[1] for x in test_set}