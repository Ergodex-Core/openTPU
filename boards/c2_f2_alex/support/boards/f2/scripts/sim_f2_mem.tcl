set run [file normalize [lindex $argv 0]]
set iprun [file normalize [lindex $argv 1]]
set pcisrun [file normalize [lindex $argv 2]]
cd $run
set_param general.maxThreads 4
create_project sim $run/project -part xcvu47p-fsvh2892-2-e -force
set_property XPM_LIBRARIES {XPM_CDC XPM_MEMORY XPM_FIFO} [current_project]
add_files $iprun/ip_project/f2_ip.srcs/sources_1/bd/f2_mem/f2_mem.bd
add_files $iprun/ip_project/f2_ip.gen/sources_1/bd/f2_mem/hdl/f2_mem_wrapper.v
read_ip $pcisrun/ip_project/pcis.srcs/sources_1/ip/f2_pcis_pipe/f2_pcis_pipe.xci
add_files -fileset sim_1 [glob $run/rtl/*.sv]
set_property top tb_f2_mem [get_filesets sim_1]
set_property xsim.simulate.runtime {all} [get_filesets sim_1]
launch_simulation -simset sim_1 -mode behavioral
close_sim
