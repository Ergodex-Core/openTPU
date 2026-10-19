set run [file normalize [lindex $argv 0]]
cd $run
create_project pcis $run/ip_project -part xcvu47p-fsvh2892-2-e -force
set_param general.maxThreads 4
create_ip -name axi_register_slice -vendor xilinx.com -library ip -module_name f2_pcis_pipe
set_property -dict [list CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {512} CONFIG.ID_WIDTH {16} CONFIG.REG_AW {1} CONFIG.REG_W {1} CONFIG.REG_B {1} CONFIG.REG_AR {1} CONFIG.REG_R {1}] [get_ips f2_pcis_pipe]
generate_target all [get_ips f2_pcis_pipe]
create_ip_run [get_files f2_pcis_pipe.xci]
launch_runs f2_pcis_pipe_synth_1 -jobs 1
wait_on_run f2_pcis_pipe_synth_1
if {[get_property PROGRESS [get_runs f2_pcis_pipe_synth_1]] ne "100%"} {error "PCIS IP failed"}
set fp [open ip_done.txt w];puts $fp IP_DONE;close $fp
