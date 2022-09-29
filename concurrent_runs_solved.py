# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:41:21 2022

@author: KEPetousakis
"""

import concurrent.futures
from time import sleep
import numpy as np
import os
import psutil 

exec_name = 'v1_pcell_solved_multithreaded'

neurons = [int(x) for x in range(0,2)]
runs = [int(x) for x in range(0,5)]

# Full range
# stims = [int(x*10) for x in range(0,10)]

# Short range
stims = [0, 30, 60, 90]

att = 2


input_data = []
nproc = np.max([1, int(len(psutil.Process().cpu_affinity())-1)])

for neuron in neurons:
	for run in runs:
		for stim in stims:
			input_data.append((neuron,run,stim))

def task(data):
	nrn = data[0]; run = data[1]; stim = data[2]
	if not os.path.exists(f'results/n{nrn}_r{run}_s{stim}_a{att}.pkl'):
		os.system(f'python {exec_name}.py nrn={nrn:.0f} run={run:.0f} stim={stim:.0f} att={att:.0f}')
	else:
		print(f'Skipping "python {exec_name}.py nrn={nrn:.0f} run={run:.0f} stim={stim:.0f} att={att:.0f}" - found results at "results/n{nrn}_r{run}_s{stim}_a{att}.pkl"')
	return f'Completed nrn={nrn:.0f} run={run:.0f} stim={stim:.0f} att={att:.0f}'

def main():
	with concurrent.futures.ProcessPoolExecutor(max_workers=nproc) as executor:
		tasklist_dict = {executor.submit(task, x): x for x in input_data}
		for future in concurrent.futures.as_completed(tasklist_dict):
			datum = tasklist_dict[future]
			try:
				status = future.result()
			except Exception as e:
				print(f'Exception {e} due to input {datum}')
			else:
				print(f'Input {datum} returned with status {status}')
				
if __name__ == '__main__':
	main()

