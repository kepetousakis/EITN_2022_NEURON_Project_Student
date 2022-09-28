# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 18:54:17 2022

@author: KEPetousakis
"""

import numpy as np
import matplotlib.pyplot as plt
import Code_General_utility_spikes_pickling as util
import os

neurons = [int(x) for x in range(0,2)]
runs = [int(x) for x in range(0,5)]

# Full range
# stims = [int(x*10) for x in range(0,10)]

# Short range
stims = [0, 30, 60, 90]

att = [0, 1, 2]

path_to_results = 'results'

time = []

if type(att) == type(0):
	data = np.zeros(shape=(len(neurons), len(runs), len(stims), 25001))
	spikes = np.zeros(shape=(len(neurons), len(runs), len(stims)))

	for neuron in neurons:
		for run in runs:
			for istim,stim in enumerate(stims):
				print(f'Processing "{path_to_results}/n{neuron}_r{run}_s{stim}_a{att}.pkl"...', end='')
				if os.path.exists(f'{path_to_results}/n{neuron}_r{run}_s{stim}_a{att}.pkl'):
					datum = util.pickle_load(f'{path_to_results}/n{neuron}_r{run}_s{stim}_a{att}.pkl')
				else:
					print('Error! Missing file: "{path_to_results}/n{neuron}_r{run}_s{stim}_a{att}.pkl"')
					raise Exception
				aux = np.array(datum[1])
				data[neuron, run, istim, :] = aux
				(nsp, times) = util.GetSpikes(aux[5000:])
				spikes[neuron, run, istim] = nsp/2
				print(f'found {nsp} spike(s) in 2s ({nsp/2:.2f} Hz)...', end='')
				if time == []:
					time = datum[0]
				print('done.')
					
	# Average across runs per neuron
	spikes = np.mean(spikes, axis=1)

	# Average across neurons, keep std
	spikes_avg = np.mean(spikes, axis=0)
	spikes_std = np.std(spikes, axis=0)

	# Plot average + std
	plt.figure()
	plt.errorbar(stims, spikes_avg, spikes_std, capsize=8, c='k')
	plt.xlabel('Stimulus orientation (deg)')
	plt.xticks(stims, stims)
	plt.ylabel('Neuronal response (Hz)')
	plt.title(f'Tuning Curve (N = {len(neurons)*len(runs)}, att={att})')
	fig = plt.gcf()
	fig.set_tight_layout(True)
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	plt.show()

	OSI = lambda r_pref,r_orth: (r_pref-r_orth)/(r_pref+r_orth)

	print(f'Analysis complete.\nPeak firing rate: {spikes_avg[0]:.2f} Hz, min firing rate {spikes_avg[-1]:.2f} Hz | OSI for this configuration (att={att}): {OSI(spikes_avg[0], spikes_avg[-1])}.')



elif type(att) == type([]):
	data = np.zeros(shape=(len(neurons), len(runs), len(stims), len(att), 25001))
	spikes = np.zeros(shape=(len(neurons), len(runs), len(stims), len(att)))
	for a in att:
		for neuron in neurons:
			for run in runs:
				for istim,stim in enumerate(stims):
					print(f'Processing "{path_to_results}/n{neuron}_r{run}_s{stim}_a{a}.pkl"...', end='')
					if os.path.exists(f'{path_to_results}/n{neuron}_r{run}_s{stim}_a{a}.pkl'):
						datum = util.pickle_load(f'{path_to_results}/n{neuron}_r{run}_s{stim}_a{a}.pkl')
					else:
						print('Error! Missing file: "{path_to_results}/n{neuron}_r{run}_s{stim}_a{a}.pkl"')
						raise Exception
					aux = np.array(datum[1])
					data[neuron, run, istim, a, :] = aux
					(nsp, times) = util.GetSpikes(aux[5000:])
					spikes[neuron, run, istim, a] = nsp/2
					print(f'found {nsp} spike(s) in 2s ({nsp/2:.2f} Hz)...', end='')
					if time == []:
						time = datum[0]
					print('done.')
					
	# Average across runs per neuron
	spikes = np.mean(spikes, axis=1)

	# Average across neurons, keep std
	spikes_avg = np.mean(spikes, axis=0)
	spikes_std = np.std(spikes, axis=0)
	
	OSI = lambda r_pref,r_orth: (r_pref-r_orth)/(r_pref+r_orth)

	# Plot average + std
	clist = ['k','r','b']
	attnstate = ['Inattentive', 'Attentive (x)', 'Attentive (+)']
	plt.figure()
	for i,a in enumerate(att):
		plt.errorbar(stims, spikes_avg[:,i], spikes_std[:,i], capsize=8, c=clist[i], label=attnstate[a])
		print(f'Analysis complete.\nPeak firing rate: {spikes_avg[0,i]:.2f} Hz, min firing rate {spikes_avg[-1,i]:.2f} Hz | OSI for this configuration ({attnstate[a]}): {OSI(spikes_avg[0,i], spikes_avg[-1,i])}.')
	plt.xlabel('Stimulus orientation (deg)')
	plt.legend()
	plt.xticks(stims, stims)
	plt.ylabel('Neuronal response (Hz)')
	plt.title(f'Tuning Curve (N = {len(neurons)*len(runs)})')
	fig = plt.gcf()
	fig.set_tight_layout(True)
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	plt.show()

	


