
import os
import re
import math
import time
import argparse
from cilp import CILP, utils
from experiments import bk
from split_folds import get_dataset
from sampling import Sampling
import pandas as pd

FACTS = 0
POS = 1
NEG = 2

TRAINING = 0
VALIDATION = 1
TESTING = 2

N_RUNS = 1

sampling = Sampling()

timestamps = {}

def main(args):

    for run in range(N_RUNS):
        print(f"Number of run {run+1}")
        for rate in [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 1]:

            if rate not in timestamps:
                timestamps[rate] = {
                    "training": [],
                    "validation": [],
                    "testing": []
                    }

            params = {
                'mlp_params': {
                    'hidden_sizes': [2],
                    'activation': 'Sigmoid'
                },
                'optim': 'Adam',
                'optim_params': {
                    'lr': 0.01,
                    'amsgrad': False
                },
                'batch_size': 32,
                'data_dir': args.data_dir,
            }

            args.rate = rate
            n_folds = args.n_splits

            source = args.data_dir.split('/')[1]

            print(f"Executing experiment for {source} and rate {args.rate}")

            #print("Training")
            src_train_facts, src_train_pos, src_train_neg = get_dataset(source, args.target, bk, n_folds, TRAINING)
            training_set = [src_train_facts, src_train_pos, src_train_neg]
            del src_train_facts, src_train_pos, src_train_neg

            #print("Validation")
            src_val_facts, src_val_pos, src_val_neg = get_dataset(source, args.target, bk, n_folds, VALIDATION)
            validation_set = [src_val_facts, src_val_pos, src_val_neg]
            del src_val_facts, src_val_pos, src_val_neg

            #print("Testing")
            src_test_facts, src_test_pos, src_test_neg = get_dataset(source, args.target, bk, n_folds, TESTING)
            test_set = [src_test_facts, src_test_pos, src_test_neg]
            del src_test_facts, src_test_pos, src_test_neg

            if args.rate < 1:
                training_set[POS] = sampling.random(training_set[POS], rate=args.rate)
                training_set[NEG] = sampling.random(training_set[NEG], rate=args.rate)

            dataset_path = f"{args.data_dir}"
            if not os.path.isdir(dataset_path + f'/train_{run+1}'): os.makedirs(dataset_path + f'/train_{run+1}')
            if not os.path.isdir(dataset_path + '/validation'): os.makedirs(dataset_path + '/validation')
            if not os.path.isdir(dataset_path + '/test'): os.makedirs(dataset_path + '/test')

            utils.write_examples(training_set[FACTS], f'{dataset_path}/train_{run+1}/bk.pl', end_of_line='')
            utils.write_examples(training_set[POS], f'{dataset_path}/train_{run+1}/pos.pl', end_of_line='')
            utils.write_examples(training_set[NEG], f'{dataset_path}/train_{run+1}/neg.pl', end_of_line='')

            utils.write_examples(validation_set[FACTS], f'{dataset_path}/validation/bk.pl', end_of_line='')
            utils.write_examples(validation_set[POS], f'{dataset_path}/validation/pos.pl', end_of_line='')
            utils.write_examples(validation_set[NEG], f'{dataset_path}/validation/neg.pl', end_of_line='')

            utils.write_examples(test_set[FACTS], f'{dataset_path}/test/bk.pl', end_of_line='')
            utils.write_examples(test_set[POS], f'{dataset_path}/test/pos.pl', end_of_line='')
            utils.write_examples(test_set[NEG], f'{dataset_path}/test/neg.pl', end_of_line='')

            start = time.time()
            model = CILP(dataset_path + f'/train_{run+1}',
                        rate=args.rate,
                        run=run+1,
                        use_semi_prop=True,
                        h_arity=2)
            
            bottom_clauses = model.init_data()
            end = time.time()
            #print(f"Time to create BCs and vectors: {end-start}")
            timestamps[rate]["training"].append(end-start)

            #sorted(training_set[FACTS])
            
            #global_vars, local_vars = [], []
            '''for posneg in ['pos', 'neg']:
                for bc in bottom_clauses[posneg]:
                    bc_sorted = []
                    for literal in bc:
                        variables = re.search(r'\((.*?)\)', literal).group(1).split(",")
                        if any(var in ["A", "B"] and var.isalpha() for var in variables):
                            global_vars.append(literal)
                        else:
                            local_vars.append(literal)
                    bc_sorted = global_vars + local_vars
                    rule = experiment['predicate'] + '(A,B) :- ' + ','.join(bc) + '.\n'
                    training_set[FACTS].append(rule)'''

            train_metrics, bnb = model.train()

            # Validation
            start = time.time()
            model = CILP(dataset_path + '/validation',
                        rate=args.rate,
                        run=run+1,
                        use_semi_prop=True,
                        h_arity=2)
            
            bottom_clauses = model.init_data()
            end = time.time()
            timestamps[rate]["validation"].append(end-start)

            validation_metrics = model.test(bnb)

            # Test
            start = time.time()
            model = CILP(dataset_path + '/test',
                        rate=args.rate,
                        run=run+1,
                        use_semi_prop=True,
                        h_arity=2)
            
            bottom_clauses = model.init_data()
            end = time.time()
            timestamps[rate]["testing"].append(end-start)

            test_metrics = model.test(bnb)

            # Save results
            log_path = f"{args.log_dir}_{run+1}"
            if not os.path.isdir(log_path + '/train'): os.makedirs(log_path + '/train')
            if not os.path.isdir(log_path + '/validation'): os.makedirs(log_path + '/validation')
            if not os.path.isdir(log_path + '/test'): os.makedirs(log_path + '/test')

            pd.Series(train_metrics).to_json(f"{log_path}/train/metrics_{args.rate}.json")
            pd.Series(validation_metrics).to_json(f"{log_path}/validation/metrics_{args.rate}.json")
            pd.Series(test_metrics).to_json(f"{log_path}/test/metrics_{args.rate}.json")
    
    pd.Series(timestamps).to_json(f"{dataset_path}/timestamps.json")
        
if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--log-dir', nargs='?', default='logs/imdb')
    PARSER.add_argument('--data-dir', nargs='?', default='datasets/imdb')
    PARSER.add_argument('--target', nargs='?', default='workedunder')
    #PARSER.add_argument('--alpha', nargs='?', default='1e-5')
    #PARSER.add_argument('--laplacian-smooth', nargs='?', default=True)
    PARSER.add_argument('--no-cache', action='store_true')
    PARSER.add_argument('--use-gpu', action='store_true')
    PARSER.add_argument('--trepan', action='store_true')
    PARSER.add_argument('--draw', action='store_true')
    PARSER.add_argument('--dedup', action='store_true')
    PARSER.add_argument('--max-epochs', type=int, default=10)
    PARSER.add_argument('--n-splits', type=int, default=5)
    PARSER.add_argument('--use-semi-prop', action='store_true') 
    PARSER.add_argument('--h-arity', type=int, default=2)
    PARSER.add_argument('--rate', type=float, default=0.01)
    #PARSER.add_argument('--test', action='store_true')
    #PARSER.add_argument('--sampling-rate', type=int, default=100) 

    ARGS = PARSER.parse_args()
    main(ARGS)