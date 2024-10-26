
from get_datasets import datasets
import numpy as np
import os

import sys
sys.path.append("..")

#verbose=True
source_balanced = False
balanced = False
SEED = 441773

def get_dataset(source, predicate, bk, n_folds, i=0):
    
    src_total_data = datasets.load(source, bk[source], seed=SEED)
    src_data = datasets.load(source, bk[source], target=predicate, balanced=source_balanced, seed=SEED)

    # Group and shuffle
    if source not in ['nell_sports', 'nell_finances', 'yago2s']:
        [src_train_facts, src_test_facts] =  datasets.get_kfold_small(i, src_data[0])
        [src_train_pos, src_test_pos] =  datasets.get_kfold_small(i, src_data[1])
        [src_train_neg, src_test_neg] =  datasets.get_kfold_small(i, src_data[2])
    else:
        [src_train_facts, src_test_facts] =  [src_data[0][0], src_data[0][0]]
        to_folds_pos = datasets.split_into_folds(src_data[1][0], n_folds=n_folds, seed=SEED)
        to_folds_neg = datasets.split_into_folds(src_data[2][0], n_folds=n_folds, seed=SEED)
        [src_train_pos, src_test_pos] =  datasets.get_kfold_small(i, to_folds_pos)
        [src_train_neg, src_test_neg] =  datasets.get_kfold_small(i, to_folds_neg)
    
    #print('Facts examples: %s' % len(src_train_facts))
    #print('Pos examples: %s' % len(src_train_pos))
    #print('Neg examples: %s\n' % len(src_train_neg[:2*len(src_train_pos)]))

    return (src_train_facts, src_train_pos, src_train_neg[:2*len(src_train_pos)])


#bk = {
#      'imdb': ['workedunder(+person,+person).',
#              'workedunder(+person,-person).',
#              'workedunder(-person,+person).',
              #'recursion_workedunder(+person,`person).',
              #'recursion_workedunder(`person,+person).',
#              'female(+person).',
#              'actor(+person).',
#              'director(+person).',
#              'movie(+movie,+person).',
#              'movie(+movie,-person).',
#              'movie(-movie,+person).',
#              'genre(+person,+genre).']}

#get_dataset({'id': '2', 'source':'imdb', 'target':'imdb', 'predicate':'workedunder', 'to_predicate':'workedunder', 'arity': 2}, bk)
