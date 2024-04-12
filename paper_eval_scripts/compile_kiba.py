import csv
import math
from lifelines.utils import concordance_index

from scipy import stats
from sklearn import metrics

ground = list(csv.reader(open("davis_kiba/kiba.csv")))[1:]
results = list(csv.reader(open("kiba_results.tsv"), delimiter="\t"))[1:]

print(results[0])

ground_truths = [float(x[-1]) for x in ground]
#ground_truths = [-math.log10(float(x[-1]) / 1e9) for x in ground]
predicted_results = [float(x[-2]) for x in results]

print(metrics.mean_squared_error(ground_truths, predicted_results))
print(concordance_index(ground_truths, predicted_results))