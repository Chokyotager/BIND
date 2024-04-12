import os
import json
import numpy as np

import math

from scipy.special import logit

from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcRIE, CalcEnrichment

result_dir = "DEKOIS_results"

results = os.listdir(result_dir)

all_data = list()

def calculate_sr_ef (predictions, ground_truth, cutoff_ratio):

    paired = [(x, y) for x, y in zip(predictions, ground_truth)]
    paired = sorted(paired, key=lambda x: x[0])
    
    base_factor = (len(ground_truth) - sum(ground_truth)) / len(ground_truth)

    cutoff_amount = round(cutoff_ratio * len(paired))
    sr = len([x for x in paired[:cutoff_amount] if x[1] == 0]) / cutoff_amount
    ef = sr /  base_factor

    return sr, ef

all_bedrocs = {

    "logits": list(),
    "ki": list(),
    "ic50": list(),
    "kd": list(),
    "ec50": list()

}

for result in results:

    current_data = json.loads(open(f"{result_dir}/{result}").read())

    predictions = current_data["predictions"]
    ground_truth = current_data["ground_truth"]
    dta = current_data["all_predictions"]

    # Logits
    paired = [(x, y) for x, y, z in zip(predictions, ground_truth, dta)]
    paired = sorted(paired, key=lambda x: x[0])

    # Flip
    paired = [[1 - x[0], 1 - x[1]] for x in paired]
    bedroc = CalcBEDROC(paired, 1, alpha=80.5)

    all_bedrocs["logits"].append(bedroc)


    # Ki
    paired = [(-z[0], y) for x, y, z in zip(predictions, ground_truth, dta)]
    paired = sorted(paired, key=lambda x: x[0])

    # Flip
    paired = [[1 - x[0], 1 - x[1]] for x in paired]
    bedroc = CalcBEDROC(paired, 1, alpha=80.5)

    all_bedrocs["ki"].append(bedroc)


    # IC50
    paired = [(-z[1], y) for x, y, z in zip(predictions, ground_truth, dta)]
    paired = sorted(paired, key=lambda x: x[0])

    # Flip
    paired = [[1 - x[0], 1 - x[1]] for x in paired]
    bedroc = CalcBEDROC(paired, 1, alpha=80.5)

    all_bedrocs["ic50"].append(bedroc)


    # Kd
    paired = [(-z[2], y) for x, y, z in zip(predictions, ground_truth, dta)]
    paired = sorted(paired, key=lambda x: x[0])

    # Flip
    paired = [[1 - x[0], 1 - x[1]] for x in paired]
    bedroc = CalcBEDROC(paired, 1, alpha=80.5)

    all_bedrocs["kd"].append(bedroc)


    # EC50
    paired = [(-z[3], y) for x, y, z in zip(predictions, ground_truth, dta)]
    paired = sorted(paired, key=lambda x: x[0])

    # Flip
    paired = [[1 - x[0], 1 - x[1]] for x in paired]
    bedroc = CalcBEDROC(paired, 1, alpha=80.5)

    all_bedrocs["ec50"].append(bedroc)


print(json.dumps(all_bedrocs))
