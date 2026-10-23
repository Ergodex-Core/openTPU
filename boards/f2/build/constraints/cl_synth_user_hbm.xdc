# Synthesis constraints of the HBM CL only (synth_cl_opentpu_hbm.tcl reads this after
# cl_synth_user.xdc; the DDR CL has no u_hbm). docs/v4/f2-timing.md.
#
# The HDK's cl_mem_hbm_wrapper takes the CL's 450 MHz reset through a lib_pipe of four stages
# (u_hbm/PIPE_HBM_AXI_RST), and the last stage drives the AXI_xx_ARESET_N pins of all 32 of the
# cl_hbm IP's ports, both stacks, along the whole bottom of SLR0: the a2af0904 placement timed
# that one flip-flop's recovery into the ports at -0.44 .. -0.89 ns over 2.3-2.8 ns of route.
# Synthesis replicates the net's driver per 8 loads here (the HDK's own idiom for a reset
# driver, cl_mem_perf's cl_synth_user.xdc: MAX_FANOUT on sync_rst_n_reg's net), so the placer
# can put a copy of the last stage beside each group of ports. The wrapper is the HDK's,
# copied by build.py: nothing of it is edited.
set_property MAX_FANOUT 8 [get_nets -of_objects [get_pins -hierarchical -filter {NAME =~ *PIPE_HBM_AXI_RST/pipe_reg[3]*/Q}]]
