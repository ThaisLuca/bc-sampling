
from collections import defaultdict
from os import path as osp

import time
import numpy as np

from .bcp import run_bcp
from .semi_prop import run_semi_prop
from .utils import get_features, load_json, pjoin
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier


class CILP:
    def __init__(self,
                 dataset,
                 rate,
                 run,
                 dedup=False,
                 cached=True,
                 use_gpu=True,
                 no_logger=False,
                 progress_bar=False,
                 use_semi_prop=True,
                 h_arity=None):
        self.dataset = dataset
        self.rate = rate
        self.run = run
        self.use_semi_prop = use_semi_prop
        self.h_arity = h_arity
        self.dedup = dedup
        self.cached = cached
        self.dedup = dedup
        # self.device = 'cuda:0' if use_gpu else 'cpu'
        self.use_gpu = use_gpu
        self.progress_bar = progress_bar
        self.n_features = 0

    def bcp(self):
        return run_bcp(self.dataset, self.rate, print_output=False)

    def semi_prop(self):
        run_semi_prop(self.dataset, self.h_arity, self.rate, print_output=False)

    def featurise(self):
        #if self.use_semi_prop:
        bc_file = pjoin(self.dataset, f'bc_filtered_{self.rate}.json') if self.rate < 1 else pjoin(self.dataset, f'bc_filtered.json')
        examples_dict = load_json(bc_file) 
        #else: 
        #examples_dict = load_json(pjoin(self.dataset, 'bc_0.1.json'))

        n_positives_examples, n_negative_examples = len(examples_dict['pos']), len(examples_dict['neg'])
        print(f"Loaded {len(examples_dict['pos'])} pos examples")
        print(f"Loaded {len(examples_dict['neg'])} neg examples")

        bcp_features = None
        bcp_examples = examples_dict['pos'] + examples_dict['neg']
        labels = np.concatenate([[1] * len(examples_dict['pos']), [0] * len(examples_dict['neg'])])
        
        feats_file = pjoin(self.dataset, f'feats_{self.rate}.npz')
        if osp.exists(feats_file) and self.cached and 'train' in self.dataset.split('/')[-1]:
            print('Featurise: Loading from cache - training')
            npzfile = np.load(feats_file)
            examples = npzfile['examples']
            bcp_features = npzfile['bcp_features']
            self.n_features = len(bcp_features)

        else:
            if self.dataset.split('/')[-1] in ['validation', 'test']:
                dir = '/'.join(self.dataset.split('/')[:2]) + f'/train_{self.run}'
                feats_file = pjoin(dir, f'feats_{self.rate}.npz')
                #print('Creating features for validation or test set')

                npzfile = np.load(feats_file)
                examples = bcp_examples
                bcp_features = npzfile['bcp_features']
                self.n_features = len(bcp_features)


            start_time = time.time()
            examples, bcp_features = get_features(bcp_examples, n_positives_examples, n_negative_examples, bcp_features)
            time_elapsed = time.time() - start_time
            print(f'Featurise done, took {time_elapsed:.1f} parsing output...')
            #examples = mrmr(examples, n_positives_examples, n_negative_examples, features_rate=0.1)
            np.savez(pjoin(self.dataset, f'feats_{self.rate}.npz'), examples=examples, bcp_features=bcp_features)

        self.bcp_features = bcp_features
        X = examples.astype(np.float32)
        y = np.expand_dims(labels, 1).astype(np.float32)

        print(f'Num examples: {X.shape[0]}')
        print(f'Num features: {X.shape[1]}')

        data = np.concatenate([y, X], axis=1)
        u_data = np.unique(data, axis=0)
        print(f'Unique: {u_data.shape[0]}')
        
        if self.dedup:
            y = u_data[:, 0:1]
            X = u_data[:, 1:]

            print(f'Num unique examples : {X.shape[0]}')

        self.X = X
        self.y = y

        #self.params['mlp_params'].update({'input_size': self.X.shape[1]})

    def init_data(self):
        self.bcp()
        self.semi_prop()
        self.featurise()

    def train(self, nb=True):

        metrics = defaultdict(list)
        #bnb = BernoulliNB()
        #bnb = ComplementNB(force_alpha=True)

        if nb:
            algo = MultinomialNB()

            start_time = time.time()
            model = algo.fit(self.X, self.y)
            time_elapsed = time.time() - start_time

            print(f'NB done, took {time_elapsed:.1f} parsing output...')

            y_pred = model.predict_proba(self.X)
        else:
            #algo = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
            algo = MLPClassifier(solver='adam', activation='tanh', hidden_layer_sizes=(self.X.shape[0]))
            #algo = MLPClassifier(solver='adam', hidden_layer_sizes=(100,))

            #[[1.]
            #[0.]
            #[0.]]

            # Parse ground truth
            # rel(A,B) == 1
            # not rel((A,B)) == 0 
            self.y = [[1, 0] if y[0] == 0 else [0, 1] for y in self.y]

            start_time = time.time()
            model = algo.fit(self.X, self.y)
            time_elapsed = time.time() - start_time

            print(f'NN done, took {time_elapsed:.1f} parsing output...')

            y_pred = model.predict_proba(self.X)
        
        metrics.update({'classes': algo.classes_})
        metrics.update({'proba': y_pred})
        #metrics.update({'score': algo.score(self.X, self.y)})

        return metrics, algo
    
    def test(self, model):
        metrics = defaultdict(list)
        #print(self.X.shape[1], self.n_features)
        
        y_pred = model.predict_proba(self.X)
        metrics.update({'classes': model.classes_})
        metrics.update({'proba': y_pred})
        #metrics.update({'score': model.score(self.X, self.y)})

        return metrics