set run [file normalize [lindex $argv 0]]
cd $run
create_project pcis $run/ip_project -part xcvu47p-fsvh2892-2-e -force
set_param general.maxThreads 4
if {[file exists $run/geometry.tcl]} {source $run/geometry.tcl} else {set otpu_S 1}
create_ip -name axi_register_slice -vendor xilinx.com -library ip -module_name f2_pcis_pipe
set_property -dict [list CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {512} CONFIG.ID_WIDTH {16} CONFIG.REG_AW {1} CONFIG.REG_W {1} CONFIG.REG_B {1} CONFIG.REG_AR {1} CONFIG.REG_R {1}] [get_ips f2_pcis_pipe]
if {$otpu_S > 1} {
  # a slice's native AXI master across an SLR (docs/v4/f2-burst200-s3.md): one per hop
  create_ip -name axi_register_slice -vendor xilinx.com -library ip -module_name f2_native_pipe
  set_property -dict [list CONFIG.ADDR_WIDTH {34} CONFIG.DATA_WIDTH {512} CONFIG.ID_WIDTH {1} CONFIG.REG_AW {1} CONFIG.REG_W {1} CONFIG.REG_B {1} CONFIG.REG_AR {1} CONFIG.REG_R {1}] [get_ips f2_native_pipe]
}
generate_target all [get_ips]
foreach ip [get_ips] {create_ip_run [get_files [get_property IP_FILE $ip]]}
set jobs [get_runs -filter {IS_SYNTHESIS && NAME != synth_1}]
launch_runs $jobs -jobs 2
foreach r $jobs {wait_on_run $r; if {[get_property PROGRESS $r] ne "100%"} {error "PCIS IP failed: $r"}}
set fp [open ip_done.txt w];puts $fp IP_DONE;close $fp
