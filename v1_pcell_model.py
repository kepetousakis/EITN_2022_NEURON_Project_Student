# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:36:19 2022

@author: KEPetousakis
"""

#%% ====================== Import Modules =====================================
# Don't forget to add modules that you'll be using.
# Examples include NumPy, pickle, seaborn (if you don't like matplotlib) etc.

from __future__ import division  
from neuron import h, gui
import matplotlib.pyplot as plt
import os

#%% ====================== Initial Setup ======================================

# Paths are relative, and rely on the file being executed from within the project folder.
h.load_file("morphology/neuron1_modified.hoc")
if os.name == 'nt':
	h.nrn_load_dll("morphology/nrnmech.dll")  # this needs to be added on Windows
h.load_file("morphology/cell_setup.hoc")

# You can inspect the morphology by using NEURON's GUI - go to "Graph" -> "Shape Plot".
# All neuronal sections are already defined above, in NEURON, this just makes them easier to access.
soma = h.soma    # 1 section
apical = h.apic  # 43 sections - access each one via "apical[x]", where "x" is the index (0-42)
basal = h.dend   # 7 sections - access each one via "basal[x]", where "x" is the index (0-6)

# Simulation parameter control - don't forget to assign these values to the actual simulation!
sim_time = 2500         # tstop for the simulation
delta = 0.1				# dt for the simulation

# Add any othe simulation parameters here, for easy access.


#%% ================== Functionality testing ================================
# Code is from Andras' tutorials, slightly tweaked. Use it simply to verify
# the model works properly. It is not required to complete the project.

func_testing = True

if func_testing:
	# Current injections
	# This dual short but intense pulse should result in an action potential.
	stim1 = h.IClamp(soma(0.5))  # add a current clamp on the middle of the soma
	stim1.delay = 500  # ms
	stim1.dur = 10 # ms
	stim1.amp = 0.33 # nA
	
	stim2 = h.IClamp(soma(0.5))  # add a second current clamp on the middle of the soma
	stim2.delay = 520  # ms
	stim2.dur = 10 # ms
	stim2.amp = 0.33 # nA
	
	# set up recordings
	soma_v = h.Vector()  # set up a recording vector
	soma_v.record(soma(0.5)._ref_v)  # record voltage at the middle of the soma
	t = h.Vector()
	t.record(h._ref_t)  # record time
	
	# run simulation
	h.v_init = -79      # Set starting potential 
	h.steps_per_ms = 1/delta  # Set simulation time resolution
	h.dt = delta        # Set simulation time resolution
	h.tstop = sim_time  # Set simulation time
	h.run();  	        # Run simulation
	
	plt.figure(figsize=(8, 5))
	plt.plot(t, soma_v, color='k', label='soma(0.5)')
	plt.xlim(0, 1000)  # Simulation is set to last 2500 ms, but we focus on the pulses.
	plt.ylim(-85,50)
	plt.xlabel('Time (ms)', fontsize=15)
	plt.ylabel('Voltage (mV)', fontsize=15)
	plt.legend(fontsize=14, frameon=False)
	plt.tight_layout()
	
#%% ================== End of Pre-Prepared Code ==============================

