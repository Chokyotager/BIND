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

main_database = json.loads(open("examples/sampled_binding_db.json").read())
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

train_dataset = {x[0]: x[1] for x in train_set}
validation_dataset = {x[0]: x[1] for x in validation_set}

train_range = [(x, y) for x, y in train_dataset.items() if y["length"] < length_cutoff]
validation_range = [(x, y) for x, y in validation_dataset.items() if y["length"] < length_cutoff]

print(len(train_range), "training datapoints loaded")
print(len(validation_range), "validation datapoints loaded")
print(len(allowed_smiles_list), "unique SMILES loaded")

def get_train_batch (amount=16, false_ratio=0.5):

    proteins = random.choices(train_range, k=amount)

    ret = list()

    for protein in proteins:

        seq_info = protein[1]

        sequence = seq_info["sequence"].upper()
        smiles = seq_info["smiles"]

        binding_ligands_set = set(all_smiles_pairs[sequence])

        is_false_ligand = False

        if random.random() < false_ratio:

            found = False
            max_tries = 100

            while not found and max_tries > 0:

                max_tries -= 1

                try:

                    random_smiles = random.choice(allowed_smiles_list)

                    if random_smiles in binding_ligands_set:
                        continue

                    smiles = random_smiles
                    found = True

                except:

                    continue

            is_false_ligand = found

        graph = smiles_graphs[smiles]

        if is_false_ligand:

            ki, ic50, kd, ec50 = [-9999] * 4

        else:

            ki, ic50, kd, ec50 = [x if x != None else -9999 for x in seq_info["log10_affinities"]]
        
        is_false_ligand = float(is_false_ligand)

        ret.append([sequence, graph, ki, ic50, kd, ec50, is_false_ligand])

    return ret

def get_validation_batch (amount=16, false_ratio=0.5):

    proteins = random.choices(validation_range, k=amount)

    ret = list()

    for protein in proteins:

        seq_info = protein[1]

        sequence = seq_info["sequence"].upper()
        smiles = seq_info["smiles"]

        binding_ligands = all_smiles_pairs[sequence]
        binding_ligands_set = set(binding_ligands)

        is_false_ligand = False

        if random.random() < false_ratio:

            found = False
            max_tries = 100

            while not found and max_tries > 0:

                max_tries -= 1

                try:

                    random_smiles = random.choice(allowed_smiles_list)

                    if random_smiles in binding_ligands_set:
                        continue

                    smiles = random_smiles
                    found = True

                except:

                    continue

            is_false_ligand = found

        graph = smiles_graphs[smiles]

        if is_false_ligand:

            ki, ic50, kd, ec50 = [-9999] * 4

        else:

            ki, ic50, kd, ec50 = [x if x != None else -9999 for x in seq_info["log10_affinities"]]
        
        is_false_ligand = float(is_false_ligand)

        ret.append([sequence, graph, ki, ic50, kd, ec50, is_false_ligand])

    return ret