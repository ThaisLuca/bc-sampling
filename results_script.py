# %%
import math
import json
from experiments import bk
from split_folds import get_dataset
from sampling import Sampling
from IPython.display import display, Markdown, Latex
from sklearn.metrics import roc_auc_score,auc,precision_recall_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
PATH = "results/"
DATASETS = {"imdb": ["workedunder", 5], "uwcse": ["advisedby", 5], "yeast": ["proteinclass", 3],
            "cora": ["samevenue", 5], "nell_sports": ["teamplayssport", 3], "nell_finances": ["companyeconomicsector", 3]} # , "twitter": ["accounttype", 3]} #, "cora": "samevenue"}

FACTS = 0
POS = 1
NEG = 2

TRAINING = 0
VALIDATION = 1
TESTING = 2

RATINGS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1]

sampling = Sampling()

# %% [markdown]
# Read files

# %%
def read_json_file(filename):
    with open(filename) as f:
        d = json.load(f)
    return d

def read_file(filename):
    with open(filename, "r") as f:
        return f.read()

# %% [markdown]
# Process timestamps files

# %%
times = {}
res = pd.DataFrame()

for dt in DATASETS:
    break

    print(dt)
    df = pd.DataFrame(read_json_file(PATH + f"logs_nn/{dt}/timestamps.json"))

    for rate in RATINGS:
        rate = '1.0' if rate == 1 else rate
        res[f'MEAN_{rate}'] = np.mean(df[f'{rate}'].tolist(), axis=1)

# %%
res

# %% [markdown]
# Process metrics files

# %%
def get_labels(resuls_array):
    results_converted = []
    for value in resuls_array['proba']:
        #if not isinstance(value,list):
        #    value = [1.0, 0]
        results_converted.append(0 if value[0] > value[1] else 1)
    return results_converted

# %%
res_AUC_ROC = {}
res_AUC_PR = {}
for source in DATASETS:

    print(f"Dataset {source}")

    if source not in res_AUC_ROC:
         res_AUC_ROC[source] = {}
         res_AUC_PR[source] = {}

    for rate in RATINGS:
            print(rate)
            
            if rate not in res_AUC_ROC[source]:

               res_AUC_ROC[source][rate] = {"train": 0, "validation": 0, "test": 0}
               res_AUC_PR[source][rate] = {"train": 0, "validation": 0, "test": 0}

            target = DATASETS[source][0]
            n_folds = DATASETS[source][1]
            #print(f"Executing analyses for {source} and rate {rate}")

            #print("Training")
            src_train_facts, src_train_pos, src_train_neg = get_dataset(source, target, bk, n_folds, TRAINING)
            training_set = [src_train_facts, src_train_pos, src_train_neg]
            #del src_train_facts, src_train_pos, src_train_neg

            #print("Validation")
            src_val_facts, src_val_pos, src_val_neg = get_dataset(source, target, bk, n_folds, VALIDATION)
            validation_set = [src_val_facts, src_val_pos, src_val_neg]
            del src_val_facts, src_val_pos, src_val_neg

            
            #print("Testing")
            src_test_facts, src_test_pos, src_test_neg = get_dataset(source, target, bk, n_folds, TESTING)
            test_set = [src_test_facts, src_test_pos, src_test_neg]
            del src_test_facts, src_test_pos, src_test_neg

            if rate < 1:
                training_set[POS] = sampling.random(training_set[POS], rate=rate)
                training_set[NEG] = sampling.random(training_set[NEG], rate=rate)

            for i in range(5):
                run = i + 1

                # Load results
                train_metrics = get_labels(read_json_file(PATH + f"logs_nn_250/{source}_{run}/train/metrics_{rate}_nn.json"))
                validation_metrics = get_labels(read_json_file(PATH + f"logs_nn_250/{source}_{run}/validation/metrics_{rate}_nn.json"))
                test_metrics = get_labels(read_json_file(PATH + f"logs_nn_250/{source}_{run}/test/metrics_{rate}_nn.json"))

                training_labels = np.concatenate([[1] * len(training_set[POS]), [0] * len(training_set[NEG])])
                validation_labels = np.concatenate([[1] * len(validation_set[POS]), [0] * len(validation_set[NEG])])
                test_labels = np.concatenate([[1] * len(test_set[POS]), [0] * len(test_set[NEG])])

                try:
                    res_AUC_ROC[source][rate]['train'] += roc_auc_score(training_labels[:len(train_metrics)], train_metrics, average="samples")
                    res_AUC_ROC[source][rate]['validation'] += roc_auc_score(validation_labels[:len(validation_metrics)], validation_metrics, average="samples")
                    res_AUC_ROC[source][rate]['test'] += roc_auc_score(test_labels[:len(test_metrics)], test_metrics, average="samples")

                except:
                    print("Something is wrong")

                precision_train, recall_train, thresholds_train = precision_recall_curve(training_labels[:len(train_metrics)], train_metrics)
                precision_validation, recall_validation, thresholds_validation = precision_recall_curve(validation_labels[:len(validation_metrics)], validation_metrics)
                precision_test, recall_test, thresholds_test = precision_recall_curve(test_labels[:len(test_metrics)], test_metrics)

                res_AUC_PR[source][rate]['train'] += auc(recall_train, precision_train)
                res_AUC_PR[source][rate]['validation'] += auc(recall_validation, precision_validation)
                res_AUC_PR[source][rate]['test'] += auc(recall_test, precision_test)
        

                del train_metrics, validation_metrics, test_metrics
                del training_labels, validation_labels, test_labels        



# %% [markdown]
# IMDB

# %%
print(pd.DataFrame(res_AUC_ROC["imdb"])/5)

# %%
print(pd.DataFrame(res_AUC_PR["imdb"])/5)

# %% [markdown]
# UWCSE

# %%
print(pd.DataFrame(res_AUC_ROC["uwcse"])/5)

# %%
print(pd.DataFrame(res_AUC_PR["uwcse"])/5)

# %% [markdown]
# Yeast

# %%
print(pd.DataFrame(res_AUC_ROC["yeast"])/5)

# %%
print(pd.DataFrame(res_AUC_PR["yeast"])/5)

# %% [markdown]
# Cora

# %%
print(pd.DataFrame(res_AUC_ROC["cora"])/5)

# %%
print(pd.DataFrame(res_AUC_PR["cora"])/5)

# %% [markdown]
# NELL Sports

# %%
print(pd.DataFrame(res_AUC_ROC["nell_sports"])/5)

# %%
print(pd.DataFrame(res_AUC_PR["nell_sports"])/5)

# %% [markdown]
# NELL Finances

# %%
print(pd.DataFrame(res_AUC_ROC["nell_finances"])/5)

# %%
print(pd.DataFrame(res_AUC_PR["nell_finances"])/5)
