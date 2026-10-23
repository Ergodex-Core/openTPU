# Frozen data from AWS cl_bb_routed.small_shell.dcp, Vivado 2025.2.
# Extracted from the completed read-only shell-methodology-audit-001 report.
# Data only: canonical NAME strings; no live or opaque Vivado objects.
# Regeneration must re-review the shell and report hashes, exact pins, and properties.
proc otpu_shell_baseline_data {} {
    return {
        shell_sha256 04a5331321e86cffc7ef740fb6132b29f179e68e2d58ccae5031c989403ecca6
        vivado_version 2025.2
        report_sha256 644b93b6f3876cf46f5bfb23038e572bb4becbcc23ced3326817374211fd407c
        report_bytes 5520187
        pin_inventory_sha256 46a550f9c99dfb0a4e1ba9df9f3c112a4d669dbf47966be3bb4e30532411a14f
        pins {
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/IPIC_IF_I/eos_int_i_reg/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/IPIC_IF_I/eos_int_i_reg}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X224Y263} BEL {SLICEL.EFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[0]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[0]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X222Y295} BEL {SLICEM.BFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[10]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[10]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X221Y284} BEL {SLICEL.FFF2} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[11]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[11]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X223Y299} BEL {SLICEL.AFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[12]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[12]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X227Y299} BEL {SLICEM.AFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[13]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[13]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X218Y301} BEL {SLICEL.AFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[14]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[14]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X224Y304} BEL {SLICEL.AFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[15]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[15]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X218Y301} BEL {SLICEL.AFF2} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[16]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[16]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X223Y299} BEL {SLICEL.CFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[17]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[17]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X223Y299} BEL {SLICEL.DFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[18]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[18]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X227Y299} BEL {SLICEM.BFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[19]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[19]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X227Y293} BEL {SLICEM.AFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[1]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[1]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X227Y293} BEL {SLICEM.BFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[20]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[20]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X224Y304} BEL {SLICEL.AFF2} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[21]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[21]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X223Y301} BEL {SLICEL.AFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[22]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[22]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X221Y301} BEL {SLICEL.EFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[23]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[23]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X224Y304} BEL {SLICEL.DFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[2]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[2]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X227Y293} BEL {SLICEM.DFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[3]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[3]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X223Y301} BEL {SLICEL.CFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[4]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[4]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X223Y301} BEL {SLICEL.DFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[5]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[5]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X221Y301} BEL {SLICEL.FFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[6]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[6]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X223Y303} BEL {SLICEL.DFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[7]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[7]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X224Y304} BEL {SLICEL.DFF2} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[8]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[8]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X221Y284} BEL {SLICEL.GFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[9]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_dataout_i_reg[9]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X227Y299} BEL {SLICEM.DFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_status_i_reg[0]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_status_i_reg[0]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X227Y298} BEL {SLICEM.HFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_status_i_reg[1]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_status_i_reg[1]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X229Y276} BEL {SLICEM.HFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_status_i_reg[2]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_status_i_reg[2]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X226Y259} BEL {SLICEL.FFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_status_i_reg[3]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_status_i_reg[3]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X229Y267} BEL {SLICEM.BFF2} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_status_i_reg[4]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_status_i_reg[4]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X224Y300} BEL {SLICEL.CFF2} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_status_i_reg[5]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_status_i_reg[5]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X224Y300} BEL {SLICEL.BFF2} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_status_i_reg[6]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_status_i_reg[6]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X227Y271} BEL {SLICEM.AFF2} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_status_i_reg[7]/C} {
                cell {SL/SH_ICAP/AXI_HWICAP/U0/ICAP_SHARED.HWICAP_CTRL_I/icap_statemachine_I1/icap_status_i_reg[7]}
                pin_properties {CLASS {pin} REF_NAME {FDRE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDRE} LOC {SLICE_X226Y259} BEL {SLICEL.EFF} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {SL/SH_ICAP/ICAP/CLK} {
                cell {SL/SH_ICAP/ICAP}
                pin_properties {CLASS {pin} REF_NAME {ICAPE3} REF_PIN_NAME {CLK} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {ICAPE3} LOC {CONFIG_SITE_X0Y0} BEL {CONFIG_SITE.ICAP_TOP} IS_SEQUENTIAL {1}}
                root {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O}
            }
            {WRAPPER/RL/HOST_PCIE4C/inst/sys_reset_in_async_rst_inst/arststages_ff_reg[0]/C} {
                cell {WRAPPER/RL/HOST_PCIE4C/inst/sys_reset_in_async_rst_inst/arststages_ff_reg[0]}
                pin_properties {CLASS {pin} REF_NAME {FDCE} REF_PIN_NAME {C} DIRECTION {IN} IS_CLOCK {1}}
                cell_properties {CLASS {cell} REF_NAME {FDCE} LOC {SLICE_X205Y171} BEL {SLICEM.DFF2} IS_SEQUENTIAL {1}}
                root {WRAPPER/RL/HOST_PCIE4C/inst/bufg_gt_sysclk/O}
            }
        }
        roots {
            {SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk/O} {CLASS {pin} REF_NAME {BUFG_GT} REF_PIN_NAME {O} DIRECTION {OUT} IS_CLOCK {0}}
            {WRAPPER/RL/HOST_PCIE4C/inst/bufg_gt_sysclk/O} {CLASS {pin} REF_NAME {BUFG_GT} REF_PIN_NAME {O} DIRECTION {OUT} IS_CLOCK {0}}
        }
    }
}
