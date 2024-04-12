import os
import json
import numpy as np

import math

from scipy.special import logit

from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcRIE, CalcEnrichment

result_dir = "CASF_results"

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


for result in results:

    current_data = json.loads(open(f"{result_dir}/{result}").read())

    predictions = current_data["predictions"]
    ground_truth = current_data["ground_truth"]
    dta = current_data["all_predictions"]

    sr_05, ef_05 = calculate_sr_ef(predictions, ground_truth, 0.005)
    sr_1, ef_1 = calculate_sr_ef(predictions, ground_truth, 0.01)

    current_data["sr_05"] = sr_05
    current_data["sr_1"] = sr_1
    #current_data["ef_05"] = ef_05

    paired = [(x, y) for x, y, z in zip(predictions, ground_truth, dta)]

    paired = sorted(paired, key=lambda x: x[0])

    # Flip
    paired = [[1 - x[0], 1 - x[1]] for x in paired]

    #print(paired)

    # Top 5
    #print(result)
    #print(sum([x[1] for x in paired[:round(len(paired) * 0.05)]]))

    bedroc = CalcBEDROC(paired, 1, alpha=80.5)
    current_data["ef_05"], current_data["ef_1"], current_data["ef_5"], current_data["ef_10"] = CalcEnrichment(paired, 1, fractions=[0.005, 0.01, 0.05, 0.1])

    rie = CalcRIE(paired, 1, alpha=80.5)

    #print(enr, current_data["ef_05"])

    current_data["bedroc"] = bedroc
    current_data["rie"] = rie

    all_data.append(current_data)

print("EFs ==========")
print("0.5%:", sum([x["ef_05"] for x in all_data]) / len(all_data))
print("1%:", sum([x["ef_1"] for x in all_data]) / len(all_data))
print("5%:", sum([x["ef_5"] for x in all_data]) / len(all_data))
print("10%:", sum([x["ef_10"] for x in all_data]) / len(all_data))

print("\nSRs ==========")
print("0.5%:", sum([x["sr_05"] for x in all_data]) / len(all_data))
print("1%:", sum([x["sr_1"] for x in all_data]) / len(all_data))
print("5%:", sum([x["sr_5"] for x in all_data]) / len(all_data))
print("10%:", sum([x["sr_10"] for x in all_data]) / len(all_data))

print("\nAUROC ==========")
print("Average", sum([x["auroc"] for x in all_data]) / len(all_data))
print("Median", np.median([x["auroc"] for x in all_data]))

print("\nBEDROC (alpha = 80.5) ==========")
print("Average", sum([x["bedroc"] for x in all_data]) / len(all_data))
print("Median", np.median([x["bedroc"] for x in all_data]))

print("\nRIE (alpha = 80.5) ==========")
print("Average", sum([x["rie"] for x in all_data]) / len(all_data))
print("Median", np.median([x["rie"] for x in all_data]))