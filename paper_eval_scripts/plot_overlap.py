import os
import json
import numpy as np

import matplotlib.pyplot as plt

from scipy.special import logit

result_dir = "CASF_results"

results = os.listdir(result_dir)

all_predictions = list()
all_ground = list()
all_special = list()

for result in results:

    current_data = json.loads(open(f"{result_dir}/{result}").read())

    predictions = current_data["predictions"]
    ground_truth = current_data["ground_truth"]
    ic50 = [x[1] for x in current_data["all_predictions"]]

    all_predictions.extend(predictions)
    all_ground.extend(ground_truth)
    all_special.extend(ic50)

def kl_divergence(p, q):

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

all_predictions = [logit(x) for x in all_predictions]

plt.figure(figsize=(5.5, 5.5))

predictions_0 = [all_predictions[i] for i in range(len(all_predictions)) if all_ground[i] == 0]
predictions_1 = [all_predictions[i] for i in range(len(all_predictions)) if all_ground[i] == 1]

plt.grid()

n_0, bins_0, _ = plt.hist(predictions_0, color='blue', alpha=0.5, label='True binder', density=True, bins=50)
n_1, bins_1, _ = plt.hist(predictions_1, color='red', alpha=0.5, label='Decoy', density=True, bins=50)

kl_div = kl_divergence(n_0, n_1)

print(kl_div)

plt.xlabel("BIND logit score")
plt.ylabel("Frequency")
plt.title("CASF")
plt.savefig("casf.png", dpi=300)