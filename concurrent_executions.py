# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:41:21 2022

@author: KEPetousakis
"""

import concurrent.futures
import sys
import os
import datetime
import numpy as np
import psutil 

# This implementation makes use of launch arguments for the Python executable.
# To do that, you need to add the following code at the top of the model (i.e. in the "v1_pcell_model.py" file).
# This code should be in a *separate* file, and not added inside the model file.

def log(text_str):
	if _LOGGING:
		with open(f'logs/log.dat', 'a+') as f:
			f.write(f'[{datetime.datetime.now()}] | {text_str}\n')
		return
	else:
		return
	
def mprint(arg):
	if _VERBOSE:
		print(arg)
		return
	else:
		return

log('Starting sim: Parsing input args.')
log(sys.argv)

_ARGLIST = ['parameter_alpha', 'parameter_beta']
_ARGVALS = {'parameter_alpha':None, 'parameter_beta':None}  # Default values - should be overwritten

try:
	if len(sys.argv) <= 1:
		log('<!> Insufficient args')
		raise Exception
		
	else:
		log('Detected runtime arguments...')
		multithreaded = True  # Use the "multithreaded" variable to control whether individual plots are shown etc.
		for i, arg in enumerate(sys.argv):
			if i > 0:
				for valid_arg in _ARGLIST:
					if valid_arg in arg:
						# raise ValueError(f'<!> {sys.argv}')
						_ARGVALS[valid_arg] = float(arg[len(valid_arg)+1:])
						log(f'Valid argument assignment: {valid_arg} = {_ARGVALS[valid_arg]:.4f}')
except Exception as e:
	log(f'Error: insufficient input args or other exception: {e}')
	log(sys.argv)
	multithreaded = False # Use the "multithreaded" variable to control whether individual plots are shown etc.

# ==== END OF EXTRA CODE FOR v1_pcell_model.py =====

# Concurrent execution code follows:

parameter_alpha = ['this', 'is', 'just', 'an', 'example']
parameter_beta = ['parameters', "don't", 'need', 'to', 'be', 'the', 'same', 'length']

experiment_mode = 0  # 0 - control / 1 - experiment (e.g. high attention, or something)


input_params = []
nproc = np.max([2, int(len(psutil.Process().cpu_affinity())-2)])  # Counts the number of available CPU cores, subtracts 2, then does np.max between that and 2.
                                                                  # You can also subtract 1, or 0 - but your computer might freeze while sims are running.

for pA in parameter_alpha
	for pB in parameter_beta:
			input_params.append((pA, pB))

def task(input_params):
	param_1 = input_params[0]; param_2 = input_params[1]
	if not os.path.exists(f'results/pA{param_1}_pB{param_2}_mode{experiment_mode}.pkl'):  # Checks whether this simulation was executed before by looking for stored results
		os.system(f'python v1_pcell_model.py parameter_alpha={param_1} parameter_beta={param_2} mode={experiment_mode}')
	else:
		print(f'Skipping "python v1_pcell_model.py parameter_alpha={param_1} parameter_beta={param_2} mode={experiment_mode}" - found results at "results/pA{param_1}_pB{param_2}_mode{experiment_mode}.pkl"')
	return f'Completed parameter_alpha={param_1} parameter_beta={param_2} mode={experiment_mode}'

def main()
	with concurrent.futures.ProcessPoolExecutor(max_workers=nproc) as executor:
		task_dict = {executor.submit(task, x): x for x in input_params}
		for future in concurrent.futures.as_completed(task_dict):
			params = task_dict[future]
			try:
				status = future.result()
			except Exception as e:
				print(f'Exception {e} due to input {params}')
			else:
				print(f'Input {params} returned with status {status}')
				
if __name__ == '__main__':
	main()

