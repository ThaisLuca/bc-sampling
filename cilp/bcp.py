import json
import math
import re
import tempfile
import time
from os import path as osp

from .utils import aleph_settings, create_script, load_examples, run_aleph, pjoin, load_json


def run_bcp(dataset, rate, cached=True, print_output=False):
    print('Running BCP')
    bc_file = pjoin(dataset, f'bc_{rate}.json') if rate < 1 else pjoin(dataset, f'bc.json')
    if osp.exists(bc_file) and cached:
        print('BCP: Loading from cache')
        if rate < 1:
            return load_json(pjoin(dataset, f'bc_{rate}.json'))
        else:
            return load_json(bc_file)

    bottom_clauses = {}
    for posneg in ['pos', 'neg']:
        bottom_clauses[posneg] = []

        train_pos = pjoin(dataset, f'{posneg}.pl')
        pos_examples = load_examples(train_pos)
        bk_file = pjoin(dataset, 'bk.pl')

        dir = '/'.join(dataset.split('/')[:2])
        mode_file = pjoin(dir, 'mode.pl')
        
        data_files={'train_pos': train_pos}

        #if test:
        #    data_files['test_pos'] = test_pos

        script_lines = aleph_settings(mode_file, bk_file, data_files=data_files)
        # script_lines += [f':- set(train_pos, "{train_pos}").']
        for i in range(len(pos_examples)):
            script_lines += [f':- sat({i+1}).']

        temp_dir = tempfile.mkdtemp()
        script_file = create_script(temp_dir, script_lines)

        print(f'Running Prolog script {script_file}')
        start_time = time.time()
        prolog_output = run_aleph(script_file)
        time_elapsed = time.time() - start_time
        print(f'Prolog done, took {time_elapsed:.1f} parsing output...')

        if print_output:
            print(prolog_output)

        bottom_clauses_raw = re.findall(r'\[bottom clause\]\n(.*?)\n\[literals\]', prolog_output,
                                        re.S)

        for b in bottom_clauses_raw:
            clause = re.sub(r'[ \n]', '', b).split(':-')
            if len(clause) == 1:
                continue
            body = clause[1]
            body = re.findall(r'(\w+\([\w,]+\))', body)
            bottom_clauses[posneg].append(sorted(body))

    with open(bc_file, 'w') as f:
        json.dump(bottom_clauses, f, indent=4)

    return bottom_clauses
