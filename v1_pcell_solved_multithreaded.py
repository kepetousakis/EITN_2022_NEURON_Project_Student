# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:36:19 2022

@author: KEPetousakis
"""

from __future__ import division
from neuron import h, gui
import matplotlib.pyplot as plt
import numpy as np
import os
import Code_General_utility_spikes_pickling as util
import datetime
import sys

_LOGGING = True
_VERBOSE = False

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

_ARGLIST = ['nrn', 'run', 'stim', 'att']
_ARGVALS = {'nrn':0, 'run':0, 'stim':0, 'att':0}

try:
	if len(sys.argv) <= 1:
		log('<!> Insufficient args')
		raise Exception
		
	else:
		log('Detected runtime arguments...')
		multithreaded = True
		for i, arg in enumerate(sys.argv):
			if i > 0:
				for valid_arg in _ARGLIST:
					if valid_arg in arg:
						# raise ValueError(f'<!> {sys.argv}')
						_ARGVALS[valid_arg] = int(arg[len(valid_arg)+1:])  # casting to integer, so be careful in case a float is needed
						log(f'Valid argument assignment: {valid_arg} = {_ARGVALS[valid_arg]:.4f}')
except Exception as e:
	log(f'Error: insufficient input args or other exception: {e}')
	log(sys.argv)
	multithreaded = False

h.load_file("morphology/neuron1_modified.hoc")
if os.name == 'nt':
	h.nrn_load_dll("morphology/nrnmech.dll")  # this needs to be added on Windows
h.load_file("morphology/cell_setup.hoc")

soma = h.soma
apical = h.apic
basal = h.dend

sim_time = 2500         # tstop for the simulation
delta = 0.1				# dt for the simulation

# ====================== Students get above code only ========================

# Control variables
bg = 1                  # Toggle background (noise) on-off
bst = 1					# Toggle basal stim-driven inputs on-off
ast = 1					# Toggle apical stim-driven inputs on-off
att = _ARGVALS['att']					# Toggle attentional inputs on-off (implemented as apical only)
atag = 0                # Sets preference of attentional inputs (can be same as stim_orient/pref, or not)
afbg = 0.25             # Percentage of apical background synapses "converted" to attentional inputs
afst = 0                # Percentage of apical stimulus-driven synapses "converted" to attentional inputs
afreq = 0.3             # Stimulation frequency of attentional inputs (for the poisson spike train)
stim_orient = _ARGVALS['stim']         # Presented stimulus orientation (mean preference of apical/basal set via "tag_apical"/"tag_basal" vars)
nrn = _ARGVALS['nrn']					# Neuron ID (alters random number generator seed)
run = _ARGVALS['run']                 # Simulation ID (alters random number generator seed)
tA = 0                  # Tag Apical
tB = 0                  # Tag Basal


#%% ================== Calculating synapses ================================
# Numbers originally from Park et al., 2019, slightly adjusted

tl = 3298.1508 						    #Total Length of Neuron
syn_exc_total = int(tl*2)				#Assume synaptic density 2/micrometer     
print(f"Total number of synapses: {syn_exc_total}")
syn_exc = int(0.75*syn_exc_total)		#25% of spines are visual responsive, Chen et al. 2013, rest set as background input.
syn_basal = int(0.6*syn_exc) 	        #deFelipe, Farinas,1998
syn_apical = int(syn_exc - syn_basal)   #deFelipe, Farinas,1998

print(f"(Background)\t Basal excitatory: {syn_basal}\t | Apical excitatory: {syn_apical}")

# Inhibitory synapses added "on top" of the excitatory ones.
syn_inh = int(syn_exc_total*0.15) 			    #15% Binzegger,Martin, 2004 (L2/3, cat V1)  
syn_inh_soma = int(syn_inh*0.07)		        #deFelipe, Farinas,1998
syn_inh_apical = int(syn_inh*0.33)   
syn_inh_basal = syn_inh - syn_inh_soma - syn_inh_apical

print(f"(Background)\t Basal inhibitory: {syn_inh_basal}\t | Apical inhibitory: {syn_inh_apical}\t | Somatic inhibitory: {syn_inh_soma}")

total_syn = int(0.25*syn_exc_total)			    #25% of spines are visual responsive, Chen et al. 2013
total_syn_basal = int(total_syn*0.6) 
total_syn_apic = total_syn - total_syn_basal

print(f"(Stimulus)\t\t Basal excitatory: {total_syn_basal}\t | Apical excitatory: {total_syn_apic}")


#%% ================== Allocating synapses ================================

# Set tuning-related parameters
apical_width = 30
basal_width = 30
tag_basal = tB
tag_apical = tA

# Set simulation-related parameters
n_neuron = nrn
n_run = run
h.steps_per_ms = int(1/delta)
h.dt = delta
h.tstop = sim_time  # set simulation time

# Set synaptic conductances
ampa_g = 0.00084
nmda_g = 0.00115
gaba_g = 0.00125

# Set random number seeds
rng_seed1 = basal_width*apical_width*(tag_basal+1)*10000*(n_run+1)*(n_neuron+1)
rng_seed2 = basal_width*apical_width*(tag_basal+1)*10000*(n_neuron+1)	

# (Approximately) Distribute background synapse numbers according to section length
apical_L = sum([x.L for x in apical])
basal_L = sum([x.L for x in basal])
apical_nsyn_bg_inh = [int(syn_inh_apical*x.L/apical_L) for x in apical]
basal_nsyn_bg_inh = [int(syn_inh_basal*x.L/basal_L) for x in basal]
somatic_nsyn_bg_inh = syn_inh - np.sum(apical_nsyn_bg_inh) - np.sum(basal_nsyn_bg_inh)
apical_nsyn_bg_exc = [int(syn_apical*x.L/apical_L) for x in apical]
basal_nsyn_bg_exc = [int(syn_basal*x.L/basal_L) for x in basal]

# (Approximately) Distribute stimulus synapse numbers according to section length
apical_nsyn_stim = [int(total_syn_apic*x.L/apical_L) for x in apical]
basal_nsyn_stim = [int(total_syn_basal*x.L/basal_L) for x in basal]

# Background excitation/inhibition parameters
freq_exc = 0.11
freq_inh = 0.11	
offset_seed = 1200

# Random number generators for position and event timings	
rng_pos = h.Random(rng_seed2)
rng_pos.uniform(0,1)
rng_t_exc = h.Random(rng_seed1+offset_seed)
rng_t_exc.poisson(freq_exc/1000)
rng_t_inh = h.Random(rng_seed1-offset_seed)
rng_t_inh.poisson(freq_inh/1000)

# Toggle background components on-off
background_enabled = bg
if background_enabled:
	basal_bg_exc_enabled = True
	basal_bg_inh_enabled = True
	apical_bg_exc_enabled = True
	apical_bg_inh_enabled = True
	somatic_bg_inh_enabled = True
else:
	basal_bg_exc_enabled = False
	basal_bg_inh_enabled = False
	apical_bg_exc_enabled = False
	apical_bg_inh_enabled = False
	somatic_bg_inh_enabled = False

#%% ================== Background synapses ================================

bg_fraction_attention_driven = afbg
bg_number_attention_driven_persection = []
bg_number_noise_driven_persection = []

# Basal excitatory background
if basal_bg_exc_enabled:
	events_bg_exc_basal = []
	vecstims_bg_exc_basal = []
	syn_bg_ampa_basal = []
	syn_bg_nmda_basal = []
	conn_bg_ampa_basal = []
	conn_bg_nmda_basal = []
	
	for i,sec in enumerate(basal):
		sec
		for isyn in range(0, basal_nsyn_bg_exc[i]):
			events_bg_exc_basal.append(h.Vector())
			vecstims_bg_exc_basal.append(h.VecStim())
			for time in range(0, int(h.tstop)):
				if rng_t_exc.repick() > 0:
					events_bg_exc_basal[-1].append(time)
			vecstims_bg_exc_basal[-1].delay = 0
			vecstims_bg_exc_basal[-1].play(events_bg_exc_basal[-1])
			syn_pos = rng_pos.repick()
			syn_bg_ampa_basal.append(h.GLU(sec(syn_pos)))
			syn_bg_nmda_basal.append(h.nmda(sec(syn_pos)))
			conn_bg_ampa_basal.append(h.NetCon(vecstims_bg_exc_basal[-1], syn_bg_ampa_basal[-1], -20, 0, ampa_g))
			conn_bg_nmda_basal.append(h.NetCon(vecstims_bg_exc_basal[-1], syn_bg_nmda_basal[-1], -20, 0, nmda_g))
	print("Allocated basal background excitatory synapses.")
			
# Basal inhibitory background
if basal_bg_inh_enabled:
	events_bg_inh_basal = []
	vecstims_bg_inh_basal = []
	syn_bg_gaba_basal = []
	conn_bg_gaba_basal = []
	
	for i,sec in enumerate(basal):
		sec
		for isyn in range(0, basal_nsyn_bg_inh[i]):
			events_bg_inh_basal.append(h.Vector())
			vecstims_bg_inh_basal.append(h.VecStim())
			for time in range(0, int(h.tstop)):
				if rng_t_inh.repick() > 0:
					events_bg_inh_basal[-1].append(time)
			vecstims_bg_inh_basal[-1].delay = 0
			vecstims_bg_inh_basal[-1].play(events_bg_inh_basal[-1])
			syn_pos = rng_pos.repick()
			syn_bg_gaba_basal.append(h.GABAa(sec(syn_pos)))
			conn_bg_gaba_basal.append(h.NetCon(vecstims_bg_inh_basal[-1], syn_bg_gaba_basal[-1], -20, 0, gaba_g))
	print("Allocated basal background inhibitory synapses.")
		
		
# Apical excitatory background
if apical_bg_exc_enabled:
	events_bg_exc_apical = []
	vecstims_bg_exc_apical = []
	syn_bg_ampa_apical = []
	syn_bg_nmda_apical = []
	conn_bg_ampa_apical = []
	conn_bg_nmda_apical = []
	
	for i,sec in enumerate(apical):
		sec
		bg_number_attention_driven_persection.append(int(apical_nsyn_bg_exc[i]*bg_fraction_attention_driven))
		bg_number_noise_driven_persection.append(apical_nsyn_bg_exc[i] - bg_number_attention_driven_persection[-1])
		print(f'\tApical {i} background excitatory synapses ({apical_nsyn_bg_exc[i]}) split: {bg_number_attention_driven_persection[-1]} attention-driven, {bg_number_noise_driven_persection[-1]} background.')
		for isyn in range(0, bg_number_noise_driven_persection[-1]):
			events_bg_exc_apical.append(h.Vector())
			vecstims_bg_exc_apical.append(h.VecStim())
			for time in range(0, int(h.tstop)):
				if rng_t_exc.repick() > 0:
					events_bg_exc_apical[-1].append(time)
			vecstims_bg_exc_apical[-1].delay = 0
			vecstims_bg_exc_apical[-1].play(events_bg_exc_apical[-1])
			syn_pos = rng_pos.repick()
			syn_bg_ampa_apical.append(h.GLU(sec(syn_pos)))
			syn_bg_nmda_apical.append(h.nmda(sec(syn_pos)))
			conn_bg_ampa_apical.append(h.NetCon(vecstims_bg_exc_apical[-1], syn_bg_ampa_apical[-1], -20, 0, ampa_g))
			conn_bg_nmda_apical.append(h.NetCon(vecstims_bg_exc_apical[-1], syn_bg_nmda_apical[-1], -20, 0, nmda_g))
	print("Allocated apical background excitatory synapses.")
	
else:
	for i,sec in enumerate(apical):
		bg_number_attention_driven_persection.append(int(apical_nsyn_bg_exc[i]*bg_fraction_attention_driven))
		bg_number_noise_driven_persection.append(apical_nsyn_bg_exc[i] - bg_number_attention_driven_persection[-1])
		
# Apical inhibitory background
if apical_bg_inh_enabled:
	events_bg_inh_apical = []
	vecstims_bg_inh_apical = []
	syn_bg_gaba_apical = []
	conn_bg_gaba_apical = []
	
	for i,sec in enumerate(apical):
		sec
		for isyn in range(0, apical_nsyn_bg_inh[i]):
			events_bg_inh_apical.append(h.Vector())
			vecstims_bg_inh_apical.append(h.VecStim())
			for time in range(0, int(h.tstop)):
				if rng_t_inh.repick() > 0:
					events_bg_inh_apical[-1].append(time)
			vecstims_bg_inh_apical[-1].delay = 0
			vecstims_bg_inh_apical[-1].play(events_bg_inh_apical[-1])
			syn_pos = rng_pos.repick()
			syn_bg_gaba_apical.append(h.GABAa(sec(syn_pos)))
			conn_bg_gaba_apical.append(h.NetCon(vecstims_bg_inh_apical[-1], syn_bg_gaba_apical[-1], -20, 0, gaba_g))
	print("Allocated apical background inhibitory synapses.")


# Somatic inhibitory background
if somatic_bg_inh_enabled:
	events_bg_inh_somatic = []
	vecstims_bg_inh_somatic = []
	syn_bg_gaba_somatic = []
	conn_bg_gaba_somatic = []
	
	soma
	for isyn in range(0, somatic_nsyn_bg_inh):
		events_bg_inh_somatic.append(h.Vector())
		vecstims_bg_inh_somatic.append(h.VecStim())
		for time in range(0, int(h.tstop)):
			if rng_t_inh.repick() > 0:
				events_bg_inh_somatic[-1].append(time)
		vecstims_bg_inh_somatic[-1].delay = 0
		vecstims_bg_inh_somatic[-1].play(events_bg_inh_somatic[-1])
		syn_pos = rng_pos.repick()
		syn_bg_gaba_somatic.append(h.GABAa(sec(syn_pos)))
		conn_bg_gaba_somatic.append(h.NetCon(vecstims_bg_inh_somatic[-1], syn_bg_gaba_somatic[-1], -20, 0, gaba_g))
	print("Allocated somatic background inhibitory synapses.")
			

#%% ================== Stim-driven synapses ================================

stim_fraction_attention_driven = afst
stim_number_attention_driven_persection = []
stim_number_stim_driven_persection = []

# Background excitation/inhibition parameters
freq_stim = 0.3
stim_rng_offset = 50000

# Toggle stim-driven excitation components on-off
basal_stim_enabled = bst
apical_stim_enabled = ast

# Emulation of different responses for each stimulus orientation via differentially weighted "tagged" synapses
presented_stimulus = stim_orient # degrees of the stimulus actually being presented
stims =  [x*10 for x in range(0,37)]  # possible stimulus degrees

# Weight vector of 36 values (37,but 37th wraps around to the 1st) that correspond to all stimulus orientations and show the proportional response (weight) to each one.
activation_vector = [0.9974,0.2487,0.0039,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0039,0.2487,0.9974,0.2487,0.0039,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0039,0.2487,0.9974]
activation_vector = [x*1.5 for x in activation_vector]

def gauss5pdf(width, x, mean):
	return (1/(width*np.sqrt(2*np.pi))) * np.sum( [ np.exp(-0.5 * ((x-mean+2*k*360)**2)/(width**2) ) for k in [-1,-0.5,0,0.5,1] ] )		
		
# Synapses are "tagged" with an orientation preference via probabilistic sampling
# so we need to initialize some random number generators.
# Pay attention to the generator used to ensure each individual basal tree section
# has the same mean orientation preference (to avoid extreme outliers), 'rng_basal_pref_mean'.
np.random.seed(int(rng_seed2))
rng_t_stim = h.Random(rng_seed1)
# We don't instantiate rng_t_stim as a poisson yet, as we need to include a value from activation_vec, which relies on
# allocation of synaptic tags (orientation preferences)

rng_pref_mean = h.Random(rng_seed2-stim_rng_offset)
rng_pref_mean.discunif(int(tag_basal/10)-int(tag_apical/10),int(tag_basal/10)+int(tag_apical/10))
rng_pos_stim = h.Random(rng_seed2+stim_rng_offset)
rng_pos_stim.uniform(0,1)


# Allocate basal stim-driven synapses
if basal_stim_enabled:
	tags_basal = []
	events_stim_basal = []
	vecstims_stim_basal = []
	syn_stim_basal = []
	syn_stim_basal = []
	conn_stim_basal = []
	conn_stim_basal = []
	# debug_stim_basal = []
	
	for i,sec in enumerate(basal):
		target_mean_preference = rng_pref_mean.repick()
		if target_mean_preference < 0:
			target_mean_preference += 180
		per_orient_probs = [gauss5pdf(basal_width,x,target_mean_preference) for x in stims]
		per_orient_norm_probs = [x/np.sum(per_orient_probs) for x in per_orient_probs]  # normalization of probabilities
		syn_tags = np.random.choice(stims, basal_nsyn_stim[i], p=per_orient_norm_probs)
		tags_basal.append(list(syn_tags))
		for j,syn in enumerate(syn_tags):
			distance_from_tag = np.abs([x-np.abs(syn-presented_stimulus) for x in stims])
			activation_idx = np.argmin(distance_from_tag)
			rng_t_stim.poisson((freq_stim*activation_vector[activation_idx])/1000)
			# debug_stim_basal.append((syn,activation_idx,activation_vector[activation_idx]))
			
			events_stim_basal.append(h.Vector())
			vecstims_stim_basal.append(h.VecStim())
			for time in range(0, int(h.tstop)):
				if rng_t_stim.repick() > 0:
					events_stim_basal[-1].append(time)
			vecstims_stim_basal[-1].delay = 500
			vecstims_stim_basal[-1].play(events_stim_basal[-1])
			syn_pos = rng_pos_stim.repick()
			syn_stim_basal.append(h.GLU(sec(syn_pos)))
			syn_stim_basal.append(h.nmda(sec(syn_pos)))
			conn_stim_basal.append(h.NetCon(vecstims_stim_basal[-1], syn_stim_basal[-1], -20, 0, ampa_g))
			conn_stim_basal.append(h.NetCon(vecstims_stim_basal[-1], syn_stim_basal[-1], -20, 0, nmda_g))
	print("Allocated basal stim-driven synapses.")

# Allocate apical stim-driven synapses
if apical_stim_enabled:
	tags_apical = []
	events_stim_apical = []
	vecstims_stim_apical = []
	syn_stim_apical = []
	syn_stim_apical = []
	conn_stim_apical = []
	conn_stim_apical = []
	
	for i,sec in enumerate(apical):
		stim_number_attention_driven_persection.append(int(apical_nsyn_stim[i]*stim_fraction_attention_driven))
		stim_number_stim_driven_persection.append(apical_nsyn_stim[i] - stim_number_attention_driven_persection[-1])
		print(f'\tApical {i} stimulus-driven synapses ({apical_nsyn_stim[i]}) split: {stim_number_attention_driven_persection[-1]} attention-driven, {stim_number_stim_driven_persection[-1]} stimulus-driven.')
		
		per_orient_probs = [gauss5pdf(apical_width,x,tag_apical) for x in stims]
		per_orient_norm_probs = [x/np.sum(per_orient_probs) for x in per_orient_probs]  # normalization of probabilities
		syn_tags = np.random.choice(stims, stim_number_stim_driven_persection[-1], p=per_orient_norm_probs) # for attention/stim split
		tags_apical.append(list(syn_tags))
		for syn in syn_tags:
			distance_from_tag = np.abs([x-np.abs(syn-presented_stimulus) for x in stims])
			activation_idx = np.argmin(distance_from_tag)
			rng_t_stim.poisson((freq_stim*activation_vector[activation_idx])/1000)
			
			events_stim_apical.append(h.Vector())
			vecstims_stim_apical.append(h.VecStim())
			for time in range(0, int(h.tstop)):
				if rng_t_stim.repick() > 0:
					events_stim_apical[-1].append(time)
			vecstims_stim_apical[-1].delay = 500
			vecstims_stim_apical[-1].play(events_stim_apical[-1])
			syn_pos = rng_pos_stim.repick()
			syn_stim_apical.append(h.GLU(sec(syn_pos)))
			syn_stim_apical.append(h.nmda(sec(syn_pos)))
			conn_stim_apical.append(h.NetCon(vecstims_stim_apical[-1], syn_stim_apical[-1], -20, 0, ampa_g))
			conn_stim_apical.append(h.NetCon(vecstims_stim_apical[-1], syn_stim_apical[-1], -20, 0, nmda_g))
	print("Allocated apical stim-driven synapses.")
	
else:
	for i,sec in enumerate(apical):
		stim_number_attention_driven_persection.append(int(apical_nsyn_stim[i]*stim_fraction_attention_driven))
		stim_number_stim_driven_persection.append(apical_nsyn_stim[i] - stim_number_attention_driven_persection[-1])
		

#%% ================== Allocate attention-driven apical synapses ============

attention_enabled = att
if att <= 1:
	gain_control_distr = [1, 0.368, 0.135, 0.05, 0.018, 0.007, 0.002, 0, 0, 0, 0,0, 0.002, 0.007,0.018, 0.05 , 0.135, 0.368, 1,0.368, 0.135, 0.05, 0.018, 0.007, 0.002, 0, 0, 0,0,0, 0.002, 0.007,0.018, 0.05 , 0.135, 0.368, 1]
elif att > 1:
	gain_control_distr = [1, 0.368, 0.135, 0.05, 0.018, 0.007, 0.002, 0, 0, 0, 0,0, 0.002, 0.007,0.018, 0.05 , 0.135, 0.368, 1,0.368, 0.135, 0.05, 0.018, 0.007, 0.002, 0, 0, 0,0,0, 0.002, 0.007,0.018, 0.05 , 0.135, 0.368, 1]
	gain_control_distr = [1 for x in gain_control_distr]
att_tag = atag

if attention_enabled:
	
	# Attention parameters
	freq_att = afreq
	att_offset_seed = 15
	
	# Random number generators for position and event timings	
	rng_pos_att = h.Random(rng_seed2-att_offset_seed)
	rng_pos_att.uniform(0,1)
	rng_t_att = h.Random(rng_seed1+att_offset_seed)
	rng_t_att.poisson(freq_att/1000)

	events_att_apical = []
	vecstims_att_apical = []
	syn_att_ampa_apical = []
	syn_att_nmda_apical = []
	conn_att_ampa_apical = []
	conn_att_nmda_apical = []
	# debug_att_events = []
	
	
	for i,sec in enumerate(apical):
		sec
		syns_for_sec = stim_number_attention_driven_persection[i]+bg_number_attention_driven_persection[i]
		print(f'\tApical {i} attention-driven synapses ({syns_for_sec}) split: {stim_number_attention_driven_persection[i]} from stim, {bg_number_attention_driven_persection[i]} from BG.')
		for isyn in range(0, syns_for_sec):
			events_att_apical.append(h.Vector())
			vecstims_att_apical.append(h.VecStim())
			# debug = []
			for time in range(0, int(h.tstop)):
				if rng_t_att.repick() > 0:
					events_att_apical[-1].append(time)
					# debug.append(time)
			# debug_att_events.append(debug)
			# print(f'\t\tPoisson train for syn {isyn} for apical {i} frequency: {len(debug_att_events[-1])} Hz.')
			vecstims_att_apical[-1].delay = 0
			vecstims_att_apical[-1].play(events_att_apical[-1])
			syn_pos = rng_pos_att.repick()
			syn_att_ampa_apical.append(h.GLU(sec(syn_pos)))
			syn_att_nmda_apical.append(h.nmda(sec(syn_pos)))
			midx = np.argmin(np.abs([x-(np.abs(att_tag-presented_stimulus)) for x in stims]))
			att_w = gain_control_distr[midx]
			conn_att_ampa_apical.append(h.NetCon(vecstims_att_apical[-1], syn_att_ampa_apical[-1], -20, 0, ampa_g*att_w))
			conn_att_nmda_apical.append(h.NetCon(vecstims_att_apical[-1], syn_att_nmda_apical[-1], -20, 0, nmda_g*att_w))
	print("Allocated apical attention-driven synapses.")
		
#%% ================== Run & Visualize =====================================	

soma_v = h.Vector()  # set up a recording vector
soma_v.record(soma(0.5)._ref_v)  # record voltage at the middle of the soma
t = h.Vector()
t.record(h._ref_t)  # record time

# run simulation
h.v_init = -79  # set starting voltage 
h.run()  # run simulation

# plt.figure(figsize=(8, 5))
# soma_v = np.array(soma_v)
# nsps, ts = util.GetSpikes(soma_v[5000:],detection_type='max')
# print(nsps, ts)
# for i,spike in enumerate(ts):
# 	plt.scatter((spike+5000)*delta, soma_v[spike+5000], s=18, c='r' )
# plt.plot(t, soma_v, color='k', label='soma(0.5)')
# plt.title(f'Neuron {nrn}, run {run}, attention {att}\nstim={stim_orient}, preference={tag_apical}|{tag_basal}, att_tag={att_tag}\nnspikes={nsps}, frequency={nsps/2}Hz')
# plt.xlim(0, sim_time)
# plt.ylim(-85, 50)
# plt.axvline(500,c='r')
# plt.xlabel('Time (ms)', fontsize=15)
# plt.ylabel('Voltage (mV)', fontsize=15)
# plt.legend(fontsize=14, frameon=False)
# plt.tight_layout()
# plt.show()

if not os.path.exists(f"results/n{_ARGVALS['nrn']}_r{_ARGVALS['run']}_s{_ARGVALS['stim']}_a{_ARGVALS['att']}.pkl"):
	util.pickle_dump((t, soma_v), f"results/n{_ARGVALS['nrn']}_r{_ARGVALS['run']}_s{_ARGVALS['stim']}_a{_ARGVALS['att']}.pkl")

print(f"Done with n{_ARGVALS['nrn']}_r{_ARGVALS['run']}_s{_ARGVALS['stim']}_a{_ARGVALS['att']}.")