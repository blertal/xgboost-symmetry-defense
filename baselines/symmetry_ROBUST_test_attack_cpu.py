############################################################
### Forked from https://github.com/cmhcbb/attackbox
############################################################

from Sign_OPT_cpu import OPT_attack_sign_SGD_cpu
from HSJA import HSJA
from OPT_attack_lf import OPT_attack_lf
from cube_attack import Cube
from models_cpu import CPUModel, CPUInvModel, CPUBothModel, XGBoostModel, XGBoostTestLoader
import os, argparse
import time
import random
from numpy import linalg as LA
import numpy as np
import json
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help='path to the config file')

attack_list = {
    "opt": OPT_attack_lf,
    "signopt": OPT_attack_sign_SGD_cpu,
    "hsja": HSJA,
    "cube": Cube,
}

args = parser.parse_args()


with open(args.config_path) as json_file:
    config = json.load(json_file)
    
print('Using config:', config)

num_attack = int(config['num_attack_per_point'])
# Cube Attack has built in support.
if config['search_mode'] == 'cube':
    num_attack = 1

print('Using num_attack:', num_attack)

test_loader = XGBoostTestLoader(args.config_path)
norm_order = config['norm_type']
# -1 was used as Inf for other benchmarks
if norm_order == -1:
    norm_order = np.inf

model = XGBoostModel(args.config_path, config['num_threads'])
#print(model.inv_model)
#exit()
amodel = CPUModel(model)
attack     = attack_list[config['search_mode']](amodel, norm_order)

def run_attack(the_attack, xi, yi, idx):
    best_norm = np.inf
    best_adv = None
    for i in range(num_attack):
        # Fix random seed for each attack.
        random.seed(8 + i)
        np.random.seed(8 + i)

        succ, adv = the_attack(xi, yi)
        if not succ:
            print('!!!Failed on example %d attack %d' % (idx, i + 1))
            continue
        current_norm = LA.norm(adv - xi, norm_order)
        ###############print('Example %d attack %d: Norm=%.4f' % (idx, i + 1, current_norm))
        if current_norm < best_norm:
            best_norm = current_norm
            best_adv = adv

    succ = best_adv is not None
    return succ, best_adv


total_Linf = 0.0
total_L1 = 0.0
total_L2 = 0.0
total_success = 0
classified_correctly = 0
inv_success = 0
both_success = 0
both_inv_success = 0

num_examples = len(test_loader)
print('Attacking %d examples...' % num_examples)
timestart = time.time()

counter = 0
correct1 = 0
correct2 = 0
adv1 = 0
adv2 = 0

for i, (xi, yi) in enumerate(test_loader):

    succ, adv = run_attack(attack, xi, yi, i + 1)


    if (amodel.predict_label(xi) == yi):
        correct1 += 1
    if (amodel.predict_label(1-xi) == yi):
        correct2 += 1

    if not succ:
        continue#adv = xi
        
    if (amodel.predict_label(xi) == yi) and (amodel.predict_label(adv) == yi):
        adv1 += 1
    if amodel.predict_label(1-adv) == yi:
        adv2 += 1

    if (amodel.predict_label(xi) != yi) or (amodel.predict_label(adv) == yi):
        continue

    counter += 1
    
    
    total_Linf += LA.norm(adv - xi, np.inf)
    total_L1 += LA.norm(adv - xi, 1)
    total_L2 += LA.norm(adv - xi, 2)

print(args.config_path + ' non-adv (' + str(correct1) + ',' + str(correct2) + ')' + ', adv (' + str(adv1) + ',' + str(adv2) + ') - robustness( norm2=' + str(total_L2) + '/' + str(counter) + ',  norminf=' + str(total_Linf) + '/' + str(counter) + ') \n\n')
    
