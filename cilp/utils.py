import collections
import hashlib
import itertools
import json
import re
import subprocess
import time
from functools import partial
from multiprocessing import Pool
from os import path as osp

import numpy as np
import pandas as pd

def get_dict_hash(d):
    return get_hash(json.dumps(d, sort_keys=True, ensure_ascii=True).encode())


def get_hash(string):
    if not isinstance(string, bytes):
        string = string.encode()
    return hashlib.sha256(string).hexdigest()[:8]


def run_aleph(script_file):
    aleph_file = get_aleph()
    cmd = f'swipl -f {aleph_file} -l {script_file}'
    return execute(cmd, return_output=True)


def get_aleph():
    curr_dir = osp.dirname(osp.realpath(__file__))
    return pjoin(curr_dir, 'aleph.pl')


def aleph_settings(mode_file, bk_file, data_files={}):
    script_lines = []
    # script_lines += [f':- set(verbosity, 0).']
    script_lines += [f':- set(depth,3).']
    for set_name, data_file in data_files.items():
        script_lines += [f':- set({set_name}, "{data_file}").']
    script_lines += [f':- read_all("{mode_file}").']
    script_lines += [f':- read_all("{bk_file}").']

    return script_lines


def create_script(directory, script_lines):
    file_name = pjoin(directory, 'script.pl')
    with open(file_name, 'w') as f:
        f.writelines([l + '\n' for l in script_lines + [':- halt.']])
    return file_name


def call(prolog, query):
    return list(prolog.query(query))


def sort_file(file_in, file_out):
    with open(file_in, 'r') as f:
        lines = f.readlines()

    with open(file_out, 'w') as f:
        f.writelines(sorted(lines))


def load_examples(file_name):
    with open(file_name, 'r') as f:
        lines = [l.strip('. \n') for l in f.readlines()]
    return np.array(lines)


def write_examples(examples, file_name, end_of_line='.'):
    with open(file_name, 'w+') as f:
        f.writelines([e + f'{end_of_line}\n' for e in examples])


def find_in_line(pattern, lines):
    found = [re.search(pattern, l) for l in lines]
    return [f.groups() for f in found if f is not None]


def load_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)


def write_json(file_name, d):
    with open(file_name, 'w') as f:
        json.dump(d, f)


# def set_feats(bcp_features, bcp_examples):
#     examples = np.zeros((len(bcp_examples), len(bcp_features)))
#     for i, ex in enumerate(bcp_examples):
#         feat_idx = np.in1d(bcp_features, ex)
#         examples[i, feat_idx] = 1
#     return examples


def set_feats(bcp_features, bcp_example):
    feat_idx = np.in1d(bcp_features, bcp_example, assume_unique=True)
    example = np.zeros((len(bcp_features)))
    example[feat_idx] = 1
    return example

def get_features(bcp_examples, n_positives, n_negatives, bcp_features=None):

    if bcp_features is None:
        bcp_features = list(set(itertools.chain.from_iterable(bcp_examples)))
        bcp_features = sorted(bcp_features)

    # set_feats_v = np.vectorize(set_feats, excluded='bcp_features')

    # examples = np.apply_along_axis(set_feats, 0, bcp_examples, bcp_features)

    start_time = time.time()
    set_feats_p = partial(set_feats, bcp_features)
    with Pool(20) as p:
        examples = p.map(set_feats_p, bcp_examples)
    examples = np.stack(examples, axis=0)

    if examples.shape[1] != len(bcp_features):
        fill_it = examples.copy()
        examples = np.zeros((examples.shape[0], len(bcp_features)))
        examples[:fill_it.shape[0],:fill_it.shape[1]] = fill_it

    print('set done, took ', time.time() - start_time)
    # examples = np.zeros((len(examples_bcp), len(bcp_features)))
    # for i, ex in enumerate(examples_bcp):
    #     feat_idx = np.in1d(bcp_features, ex)
    #     examples[i, feat_idx] = 1

    #examples = apply_laplacian_smooth(examples,n_positives,n_negatives)
    #examples = examples.drop(columns=['label'])
    #examples = examples.to_numpy()

    return examples, bcp_features


def execute(cmd, return_output=False):
    popen = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             shell=True,
                             universal_newlines=True)

    if return_output:
        output, err = popen.communicate()
    else:
        for stdout_line in iter(popen.stdout.readline, ""):
            # print(stdout_line.rstrip())
            print(stdout_line.rstrip())
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        print('Subproc error:')
        print(output)
        raise subprocess.CalledProcessError(return_code, cmd)

    if return_output:
        return output


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            v_list = {str(i): v_ for i, v_ in enumerate(v)}
            items.extend(flatten_dict(v_list, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def pjoin(root, file_name):
    # for Windows compatibility, otherwise pjoin mixes slashes
    if '/' in root:
        if '\\' in root:
            raise RuntimeError(f'Path has mixed delimiters: {root}')

        if root[-1] == '/':
            return root + file_name
        return root + '/' + file_name

    if root[-1] == '\\':
        return root + file_name
    return root + '\\' + file_name
