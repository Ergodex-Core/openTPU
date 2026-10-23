#!/usr/bin/env python3
"""Generate the isolated C2 wrapper using Alex's burst200 shell interfaces.

Derived from Alex archive revision 06c169cbcd1721ae8b167de992a93461697a397d.
The original native adapter and geometry RTL remain frozen. Eight direct HBM
ports replace the original two-port topology; this is a new physical design,
not a claim that Alex's historical timing or throughput carries over.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

PORTS = 8
CHANNEL_BYTES = 1 << 29
BEAT_BYTES = 64
CORE_KHZ = 200000
ARCHIVE_COMMIT = "06c169cbcd1721ae8b167de992a93461697a397d"


def memory_contract():
    """Machine-readable expectations for generated-IP/snapshot validation."""
    return {
        "schema": 1, "native_ports": PORTS, "physical_ports": list(range(PORTS)),
        "channel_bytes": CHANNEL_BYTES, "beat_bytes": BEAT_BYTES,
        "total_bytes": PORTS * CHANNEL_BYTES, "num_si": PORTS + 1, "num_mi": PORTS,
        "clocks_hz": {"core_clk": 200000000, "host_clk": 250000000, "hbm_clk": 450000000},
        "host": {"name": "S_HOST", "pin": "S00_AXI", "data_width": 512,
                 "address_width": 64, "id_width": 16},
        "native": [{"name": f"S_N{p}", "pin": f"S{p+1:02d}_AXI", "data_width": 512,
                    "address_width": 34, "id_width": 1, "base": p * CHANNEL_BYTES,
                    "bytes": CHANNEL_BYTES, "destination": f"M_H{p}"} for p in range(PORTS)],
        "hbm": [{"name": f"M_H{p}", "pin": f"M{p:02d}_AXI", "data_width": 256,
                 "address_width": 34, "protocol": "AXI3"} for p in range(PORTS)],
        "native_read_burst": 16, "native_max_outstanding": 128, "kernel_receive_depth": 32,
    }


def render_memory_ip():
    return Path(__file__).with_name("create_f2_ip.tcl").read_text()


def render_top(hdk: Path, build_id: str):
    """Render only; never execute imported source or modify the installed HDK.

    ``hdk`` is the AWS HDK directory ending in /hdk, as in Alex's scripts.
    Header text is used solely to identify unused shell output pins.
    """
    if not re.fullmatch(r"[0-9a-fA-F]{8}", build_id):
        raise ValueError("build_id must be exactly eight hexadecimal digits")
    build_id = build_id.lower()
    iface = Path(hdk) / "common/shell_stable/design/interfaces"
    header = (iface / "cl_ports.vh").read_text()
    includes = ("unused_ddr_template.inc", "unused_cl_sda_template.inc",
                "unused_flr_template.inc", "unused_apppf_irq_template.inc")
    unused = {name: (iface / name).read_text() for name in includes}
    lines = [f"// C2 generated shell, derived from Alex {ARCHIVE_COMMIT}.",
             "// Eight direct 512 MiB ports; requires independent physical qualification.",
             'module cl_opentpu (\n`include "cl_ports.vh"\n);']
    used = set()

    def line(text):
        lines.append(text)

    def decl(name, width=1, suffix=""):
        line("logic " + (f"[{width-1}:0] " if width > 1 else "") + name + suffix + ";")

    def inst(module, name, connections, params=None):
        for value in connections.values():
            used.update(re.findall(r"\b\w+\b", value))
        prefix = module
        if params:
            prefix += " #(\n  " + ",\n  ".join(f".{k}({v})" for k, v in params.items()) + "\n)"
        line(prefix + " " + name + " (\n  " +
             ",\n  ".join(f".{k}({v})" for k, v in connections.items()) + "\n);")

    line("logic core_clk, tmem_clk, hbm_clk, core_locked, hbm_locked;\n"
         "logic core_rst_n, hbm_rst_n, rst_main_n_sync, core_rst;\n"
         "logic [2:0] leds; logic hbm_ready, hbm_ready_core;")
    inst("f2_core_clk", "u_core_clk", dict(clk_in1="clk_main_a0", reset="!rst_main_n",
         locked="core_locked", clk_out1="core_clk", clk_out2="tmem_clk"))
    inst("f2_hbm_clk", "u_hbm_clk", dict(clk_in1="clk_hbm_ref", reset="!rst_main_n",
         locked="hbm_locked", clk_out1="hbm_clk"))
    for name, clk, source, output in [
        ("main", "clk_main_a0", "rst_main_n", "rst_main_n_sync"),
        ("core", "core_clk", "rst_main_n && core_locked && hbm_locked", "core_rst_n"),
        ("hbm", "hbm_clk", "rst_main_n && hbm_locked", "hbm_rst_n"),
    ]:
        inst("xpm_cdc_async_rst", "u_rst_" + name,
             dict(src_arst=source, dest_clk=clk, dest_arst=output),
             dict(DEST_SYNC_FF=4, RST_ACTIVE_HIGH=0))
    line("always_ff @(posedge core_clk) core_rst <= !core_rst_n;")
    inst("xpm_cdc_single", "u_ready_cdc", dict(src_clk="clk_main_a0", src_in="hbm_ready",
         dest_clk="core_clk", dest_out="hbm_ready_core"), dict(DEST_SYNC_FF=4, SRC_INPUT_REG=0))

    axil = dict(awaddr=32, awvalid=1, awready=1, wdata=32, wstrb=4, wvalid=1,
                wready=1, bresp=2, bvalid=1, bready=1, araddr=32, arvalid=1,
                arready=1, rdata=32, rresp=2, rvalid=1, rready=1)
    responses = {"awready", "wready", "bresp", "bvalid", "arready", "rdata", "rresp", "rvalid"}
    for key, width in axil.items():
        decl("ctl_" + key, width)
    connections = dict(s_axi_aclk="clk_main_a0", s_axi_aresetn="rst_main_n_sync",
                       m_axi_aclk="core_clk", m_axi_aresetn="core_rst_n")
    for key in axil:
        connections["s_axi_" + key] = ("cl_ocl_" if key in responses else "ocl_cl_") + key
        connections["m_axi_" + key] = "ctl_" + key
    connections.update(s_axi_awprot="3'd0", s_axi_arprot="3'd0", m_axi_awprot="", m_axi_arprot="")
    inst("f2_ocl_cdc", "u_ocl_cdc", connections)

    native = dict(cvalid=1, cready=1, cwe=1, caddr=25, wvalid=1, wready=1,
                  wdata=512, wmask=64, rvalid=1, rdata=512, wdone=16, err=1)
    for key, width in native.items():
        line("logic [7:0]" + (f"[{width-1}:0]" if width > 1 else "") + " n_" + key + ";")
    connections = dict(clk="core_clk", rst="core_rst", hbm_ready="hbm_ready_core", led="leds")
    for key in axil:
        connections["s_ctl_" + key] = "ctl_" + key + ("[11:0]" if key in ("araddr", "awaddr") else "")
    connections.update({"n_" + key: "n_" + key for key in native})
    inst("otpu_c2_hbm_kernel", "u_board", connections,
         dict(BUILD_ID=f"32'h{build_id}", CORE_KHZ=CORE_KHZ, DEPTH=256, MAX_ROWS=512, RD=32))

    axi = dict(awaddr=34, awvalid=1, awready=1, wdata=512, wstrb=64, wvalid=1,
               wready=1, bvalid=1, bresp=2, bready=1, araddr=34, arlen=8,
               arvalid=1, arready=1, rdata=512, rvalid=1, rlast=1, rresp=2, rready=1)
    for port in range(PORTS):
        for key, width in axi.items():
            decl(f"n{port}_" + key, width)
        for channel in ("aw", "ar"):
            decl(f"n{port}_{channel}addr_local", 34)
            # C2 restricts the native offset to 23 beats-address bits. Applying
            # the physical window here keeps Alex's BASE_ADDR=0 adapter intact.
            line(f"assign n{port}_{channel}addr = n{port}_{channel}addr_local | 34'h{port*CHANNEL_BYTES:09x};")
        connections = dict(clk="core_clk", rst="core_rst")
        connections.update({"n_" + key: f"n_{key}[{port}]" for key in native})
        connections.update({"m_" + key: f"n{port}_" + key + ("_local" if key in ("araddr", "awaddr") else "")
                            for key in axi})
        inst("otpu_native_axi", "u_native" + str(port), connections, dict(BASE_ADDR="34'd0"))
        line("`ifndef SYNTHESIS\n"
             f"always @(posedge core_clk) if(!core_rst && n_cvalid[{port}] && n_caddr[{port}][24:23]!=0)\n"
             f'  $fatal(1,"C2 native port {port}: address exceeds 512 MiB window");\n'
             "`endif")

    hbm = dict(araddr=34, arburst=2, arid=6, arlen=4, arsize=3, arvalid=1,
               awaddr=34, awburst=2, awid=6, awlen=4, awsize=3, awvalid=1,
               rready=1, bready=1, wdata=256, wlast=1, wstrb=32, wvalid=1,
               arready=1, awready=1, rdata=256, rid=6, rlast=1, rresp=2,
               rvalid=1, wready=1, bid=6, bresp=2, bvalid=1)
    hbm_outputs = {"arready", "awready", "rdata", "rid", "rlast", "rresp", "rvalid", "wready", "bid", "bresp", "bvalid"}
    for key, width in hbm.items():
        decl("h_" + key, width, " [0:7]")
    for key in ("awaddr", "araddr"):
        decl("hb_" + key, 34, " [0:7]")
    for port in range(PORTS):
        inst("otpu_hbm_geometry", "u_geometry" + str(port),
             dict(s_awaddr=f"h_awaddr[{port}]", s_araddr=f"h_araddr[{port}]",
                  m_awaddr=f"hb_awaddr[{port}]", m_araddr=f"hb_araddr[{port}]"))
    line("cfg_bus_t hbm_stat();\nassign hbm_stat.wr=1'b0;\nassign hbm_stat.rd=1'b0;\n"
         "assign hbm_stat.addr='0;\nassign hbm_stat.wdata='0;\nassign hbm_stat.user='0;")
    connections = dict(apb_clk="clk_hbm_ref", i_clk_250m="clk_main_a0", i_rst_250m_n="rst_main_n_sync",
                       i_clk_450m="hbm_clk", i_rst_450m_n="hbm_rst_n", hbm_stat_bus="hbm_stat",
                       o_cl_sh_hbm_stat_int="", o_hbm_ready="hbm_ready")
    for key in hbm:
        connections[("o" if key in hbm_outputs else "i") + "_axi_" + key] = (
            "hb_" if key in ("awaddr", "araddr") else "h_") + key
    for stack in range(2):
        for key in ("preset_n", "paddr", "pprot", "psel", "penable", "pwrite", "pwdata", "pstrb", "pready", "prdata", "pslverr"):
            connections[("i" if key == "preset_n" else "o") + f"_hbm_apb_{key}_{stack}"] = f"hbm_apb_{key}_{stack}"
    inst("cl_mem_hbm_wrapper", "u_hbm", connections, dict(NUM_OF_AXI_PORTS=8, AXI4_INTERFACE=0))

    connections = dict(core_clk="core_clk", host_clk="clk_main_a0", hbm_clk="hbm_clk", aresetn="core_rst_n")
    all_axi = list(hbm) + ["awcache", "awlock", "awprot", "awqos", "arcache", "arlock", "arprot", "arqos"]
    host_width = dict(hbm, awaddr=64, araddr=64, awid=16, arid=16, bid=16, rid=16,
                      awlen=8, arlen=8, wdata=512, rdata=512, wstrb=64,
                      awcache=4, awlock=1, awprot=3, awqos=4, arcache=4, arlock=1, arprot=3, arqos=4)
    pipe = dict(aclk="clk_main_a0", aresetn="rst_main_n_sync", s_axi_awregion="4'd0",
                s_axi_arregion="4'd0", m_axi_awregion="", m_axi_arregion="")
    for key in all_axi:
        decl("pcis_" + key, host_width[key])
        pipe["s_axi_" + key] = ("cl_sh_dma_pcis_" if key in hbm_outputs else "sh_cl_dma_pcis_") + key
        pipe["m_axi_" + key] = "pcis_" + key
        connections["S_HOST_" + key] = "pcis_" + key
    inst("f2_pcis_pipe", "u_pcis_pipe", pipe)
    for port in range(PORTS):
        for key in all_axi:
            if key in axi:
                value = f"n{port}_" + key
            elif key in ("arid", "awid"):
                value = "1'b0"
            elif key in ("rid", "bid"):
                value = ""
            elif key in ("arlen", "awlen"):
                value = "8'd0"
            elif key in ("arburst", "awburst"):
                value = "2'b01"
            elif key in ("arsize", "awsize"):
                value = "3'd6"
            elif key == "wlast":
                value = "1'b1"
            elif key in ("awcache", "arcache"):
                value = "4'b0000"  # Device non-bufferable: B means visibility.
            else:
                value = "'0"
            connections[f"S_N{port}_" + key] = value
        for key in all_axi:
            if key not in ("arid", "awid", "rid", "bid"):
                connections[f"M_H{port}_" + key] = f"h_{key}[{port}]" if key in hbm else ""
        line(f"assign h_arid[{port}] = 6'd0;\nassign h_awid[{port}] = 6'd0;")
    inst("f2_mem_wrapper", "u_mem", connections)
    for name, contents in unused.items():
        line('`include "' + name + '"')
        used.update(re.findall(r"\b\w+\b", contents))
    decl("shell_status", 11)
    inst("xpm_cdc_array_single", "u_status_cdc", dict(src_clk="core_clk", src_in="{n_err,leds}",
         dest_clk="clk_main_a0", dest_out="shell_status"),
         dict(WIDTH=11, DEST_SYNC_FF=4, SRC_INPUT_REG=1, INIT_SYNC_FF=1, SIM_ASSERT_CHK=0))
    constants = {"cl_sh_id0": "32'hf0011d0f", "cl_sh_id1": "32'h1d51fedd",
                 "cl_sh_status0": "32'h43324842", "cl_sh_status1": f"32'h{build_id}",
                 "cl_sh_status2": "{24'd0,shell_status[10:3]}",
                 "cl_sh_status_vled": "{12'd0,hbm_ready,shell_status[2:0]}"}
    outputs = re.findall(r"output\s+logic\s*(?:\[[^]]+\]\s*)?(\w+)", header)
    if len(outputs) != len(set(outputs)) or not constants.keys() <= set(outputs):
        raise ValueError("Unexpected HDK shell output declarations")
    for name in outputs:
        if name not in used:
            line("assign " + name + " = " + constants.get(name, "'0") + ";")
    line("endmodule")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--ip-script", type=Path)
    args = parser.parse_args()
    top = render_top(args.hdk, args.build_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(top)
    if args.ip_script:
        args.ip_script.parent.mkdir(parents=True, exist_ok=True)
        args.ip_script.write_text(render_memory_ip())


if __name__ == "__main__":
    main()
