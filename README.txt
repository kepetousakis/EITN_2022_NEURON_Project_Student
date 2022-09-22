Code was created and tested on Windows 10 (64bit), and was also tested on the latest MacOS and Ubuntu.
In case of any issues, contact me ("Konstantinos-Evangelos Petousakis", aka "Kostas"). 
My e-mail address is kepetousakis[at]gmail.com, in case you need to contact me via e-mail.
Send me an e-mail so I can add you to the project Slack group ("EITN_2022_nrnproject")!


INSTRUCTIONS:

> Compile the .mod files in the "morphology" folder. 
    - https://www.neuron.yale.edu/neuron/static/docs/nmodl/unix.html  for unix-based OS
    - https://www.neuron.yale.edu/neuron/static/docs/nmodl/mswin.html  for windows-based OS

> The "morphology" folder contains files that generate the reconstructed neuronal morphology and also allocate all non-synaptic 
  mechanisms required.
	- You can browse the files therein if you want, but they are written in pure NEURON, not NEURON/Python, so be warned!
	- You should not have to tweak the files in the "morphology" folder in any way. If you think you should, let me know!
	- The folder also contains files that model synaptic mechanisms ('ampa.mod', 'NMDA_syn.mod', 'gabaa.mod'), as well
	  as a file that can be used to "play back" pre-generated inputs for synapses ('vecstim.mod'). You can make use of all
	  of these files, or none of them - it is up to you. They are provided for your convenience, if you want to use them.
	  Syntax (python/NEURON pseudocode):
  		===========================================================================================================================
		  section_name                                  # to access a specific section (the target of your synapses)
		  my_synapse_ampa = h.GLU(synapse_position)     # allocates a single AMPA synapse at "synapse_position" (0-1)
		  my_synapse_nmda = h.nmda(synapse_position)    # allocates a single NMDA synapse at "synapse_position" (0-1)
		  my_synapse_gabaa= h.GABAa(synapse_position)   # allocates a single GABAa synapse at "synapse_position" (0-1)
		  my_vecstim = h.VecStim()                      # creates a VecStim object
	  	  my_vecstim.delay = 0                          # no delay to event playback (they start when simulation starts)
	  	  # my_vecstim.delay = 100                      # OR events start 100ms after simulation starts
	  	  my_vecstim.play(spike_train)                  # add events to the VecStim to play back ('spike_train' contains
	  	  												# the timing of events in ms, and can be created in many ways)
	  	  synaptic_threshold=-20                        # If spike_train[x] > synaptic_threshold, the synapse is activated
	  	  synaptic_delay=0                              # Synapses should not have an activation delay (ms)
	  	  synaptic_weight=0.001                         # Conductance of synaptic mechanisms - can have separate vars, of course

	  	  # Connect synapses with (in this case) the same event source (VecStim object) - can also create different sources
		  my_connection_ampa = h.NetCon(my_vecstim, my_synapse_ampa, synaptic_threshold, synaptic_delay, synaptic_weight)
		  my_connection_nmda = h.NetCon(my_vecstim, my_synapse_nmda, synaptic_threshold, synaptic_delay, synaptic_weight*1.1)
		  my_connection_gabaa = h.NetCon(my_vecstim, my_synapse_nmda, synaptic_threshold, synaptic_delay, synaptic_weight*1.2)
		===========================================================================================================================

> Verify that your copy of the model functions as intended by running the model file (v1_pcell_model.py). 
	- The code in the file already contains a simple functionality test with a dual pulse current injection, intended to generate
	  an action potential, shown in a simple plot.
	- This verification code is not required to continue the project.

> Move towards achieving the goals of the project as stated:
	- Main Question: "Can dendrite-bound attentional signals modulate the activity of orientation selective neurons?"

	Milestones:
	- Allocate a single synapse and ensure correct function
	- Allocate multiple synapses according to a set plan 
	- Ensure that stimulus-driven synapses feature orientation selectivity
	- Implement a subset of synapses as attentional (feedback) inputs
	- Allocate all synapses (feedforward & feedback)
	- Show that the neuron exhibits orientation tuning (tuning curve/OSI)
	- Investigate the effect of attention on neuronal output
	- Demonstrate the effect (or lack thereof) of attention
	- Is there something else you'd like to check using the model? You may do so at your discretion!
	- Present your results!


Useful Papers:
1. Silver, R. A. (2010). Neuronal arithmetic. Nature Reviews Neuroscience, 11(7), 474-489.
2. Goetz, L., Roth, A., & Häusser, M. (2021). Active dendrites enable strong but sparse inputs to determine orientation selectivity. Proceedings of the National Academy of Sciences, 118(30), e2017339118.
3. Park, J., Papoutsi, A., Ash, R. T., Marin, M. A., Poirazi, P., & Smirnakis, S. M. (2019). Contribution of apical and basal dendrites to orientation encoding in mouse V1 L2/3 pyramidal neurons. Nature Communications, 10(1), 1-11.

Other Resources:
The project presentation (included as .pdf, and also as .pptx with speaker notes included)
The NEURON tutorials by András Ecker
https://neuron.yale.edu/neuron/static/py_doc/index.html  [NEURON/Python documentation]
https://docs.python.org/3/reference/  [Python documentation]
