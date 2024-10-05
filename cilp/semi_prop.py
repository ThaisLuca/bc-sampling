import re
import json
from os import path as osp
from .utils import load_json, pjoin

def get_literals(clause: str) -> list:
    '''
        Split literals in the body of the bottom clause given as parameter
    '''
    return re.split(',\s*(?![^()]*\))', clause.split(':-')[1].strip())

def get_list_of_predicates(clause: list) -> list:
    '''
        Return string of predicates
    '''
    return re.sub(r'\([^)]*\)', '', clause)

def get_vars(literal: str) -> list:
    '''
        Split and returns all variables from a literal
    '''
    return [var for var in re.search('\(([\w\d][,\w\d]*)\)', literal).group(1).split(',') if var.istitle()]

def update_local_variables(vars: list, global_variables: list) -> list:
    '''
        Returns all variables from a literal that are NOT global variables
    '''
    return list(set(vars) - set(global_variables))

def is_redudant(features: list, rule: str, global_variables: list) -> list:
    '''
        Check if BC is redudant
    '''
    #rule_literals
    #for f in features:
        

    return 0

def build_first_order_features(rules: list, global_variables: list, features: dict = {}) -> list:
    '''
        Returns a list of first-order features
    '''

    semi_bc = []

    #print(rules)
    
    str_global_variables = ','.join(global_variables)
    for rule in rules:
        str_rule = ','.join(rule) # and not is_redudant(features, rule, global_variables)
        if str_rule not in features :
            features[str_rule] = f"L_{str(len(features) + 1)}({str_global_variables})"
        semi_bc.append(features[str_rule])
    return semi_bc, features


def run_semi_prop(data_dir: str, arity: int, rate:float, cached=True, print_output=False) -> list:
    print('Semi-Prop: Running Semi-Prop')
    bc_file = pjoin(data_dir, f'bc_filtered_{rate}.json') if rate < 1 else pjoin(data_dir, f'bc_filtered.json')
    features_file = pjoin(data_dir, 'features.json')
    if osp.exists(bc_file) and cached:
        print('Loading from cache')
        return 
    else:
        file = pjoin(data_dir, f'bc_{rate}.json') if rate < 1 else pjoin(data_dir, f'bc.json')
        bottom_clauses = load_json(file)

    global_variables = list(map(chr, range(ord('A'), ord('A')+arity)))
    
    f_bc, features = {}, {}

    for posneg in ["pos", "neg"]:
        f_bc[posneg] = []
        rules = []
        #features[posneg] = []

        for bc in bottom_clauses[posneg]:
            
            local_variables = []
            current_rule = []
            rules = []

            #literals = get_literals(bc)
            literals = bc[:]

            i = 0
            while i < len(literals):

                vars = get_vars(literals[i])

                allGlobalVars = all([var in global_variables for var in vars])

                canBeSharedVariables = [var for var in vars if var not in global_variables]
                sharesVariables = any([var in local_variables for var in canBeSharedVariables])

                if allGlobalVars or (not local_variables and not current_rule) or sharesVariables:
                    current_rule.append(literals[i])
                    local_variables += update_local_variables(vars, global_variables)
                    local_variables = list(set(local_variables))
                    literals.pop(i)
                else:
                    i += 1

                if i >= len(literals):
                    rules.append(current_rule)
                    current_rule = []
                    local_variables = []
                    i = 0
            
            new_rule, features = build_first_order_features(rules, global_variables, features)
            f_bc[posneg].append(new_rule)
            del new_rule
    
    with open(bc_file, 'w') as f:
        json.dump(f_bc, f, indent=4)
    
    with open(features_file, 'w') as f:
        json.dump(features, f, indent=4)
    print('Finished')