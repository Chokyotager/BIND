import json
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

import math

bindingdb_results = json.loads(open("bindingdb_results.tsv").read())

predicted_ki = [x["results"]["ki"] for x in bindingdb_results]
predicted_ic50 = [x["results"]["ic50"] for x in bindingdb_results]
predicted_kd = [x["results"]["kd"] for x in bindingdb_results]
predicted_ec50 = [x["results"]["ec50"] for x in bindingdb_results]

ground_ki = [x["log10_affinities"][0] for x in bindingdb_results]
ground_ic50 = [x["log10_affinities"][1] for x in bindingdb_results]
ground_kd = [x["log10_affinities"][2] for x in bindingdb_results]
ground_ec50 = [x["log10_affinities"][3] for x in bindingdb_results]

predicted_ki = [x for x, y in zip(predicted_ki, ground_ki) if y != None]
ground_ki = [x for x in ground_ki if x != None]

predicted_ic50 = [x for x, y in zip(predicted_ic50, ground_ic50) if y != None]
ground_ic50 = [x for x in ground_ic50 if x != None]

predicted_kd = [x for x, y in zip(predicted_kd, ground_kd) if y != None]
ground_kd = [x for x in ground_kd if x != None]

predicted_ec50 = [x for x, y in zip(predicted_ec50, ground_ec50) if y != None]
ground_ec50 = [x for x in ground_ec50 if x != None]

print("R^2 Ki", pearsonr(predicted_ki, ground_ki))
print("R^2 IC50", pearsonr(predicted_ic50, ground_ic50))
print("R^2 Kd", pearsonr(predicted_kd, ground_kd))
print("R^2 EC50", pearsonr(predicted_ec50, ground_ec50))

print("RMSE Ki", math.sqrt(mean_squared_error(predicted_ki, ground_ki)))
print("RMSE IC50", math.sqrt(mean_squared_error(predicted_ec50, ground_ec50)))
print("RMSE Kd", math.sqrt(mean_squared_error(predicted_kd, ground_kd)))
print("RMSE EC50", math.sqrt(mean_squared_error(predicted_ic50, ground_ic50)))