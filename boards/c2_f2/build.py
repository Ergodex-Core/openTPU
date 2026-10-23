#!/usr/bin/env python3
"""Build the isolated C2 HBM kernel against frozen F2 shell support.

No production TPU opcode/compiler is changed. Each run freezes all HDL, constraints,
tool arguments and hashes before invoking Vivado. Packaging requires route qualification.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SUPPORT = HERE / "support"
PART = "xcvu47p-fsvh2892-2-e"
CORE_SOURCES = ["vpu/otpu_fp.sv", "vpu/otpu_fpipe.sv", "top/otpu_pkg.sv",
                "tech/xilinxus/otpu_tech_dsp.sv", "mxu/otpu_mxu.sv",
                "mxu/otpu_c2_mxu.sv", "c2/otpu_c2_hbm_kernel.sv"]
FABRIC_SOURCES = ["shell/otpu_afifo.sv", "shell/otpu_link.sv", "shell/otpu_skidbuf.sv",
                  "boards/f2/otpu_axil_cdc.sv", "boards/f2/otpu_native_axi.sv",
                  "boards/f2/otpu_f2_hbm_host.sv", "boards/f2/otpu_f2_hbm_port.sv",
                  "boards/f2/otpu_f2_hbm_mem.sv"]
TUNING = {"place": "ExtraNetDelay_high", "phys": "AggressiveExplore",
          "route": "Explore", "retime": True}
# Generated products inventoried on opentpu-f2, independently downloaded and
# SHA-checked against its read-only SSM inventory on 2026-10-19.
SMARTCONNECT_SHA256 = {
    "cl_axi_sc_1x1.bd": "af85051845af4743b3381be1cb65b089ab05a6dc9f1f8ac970492223ac802d89",
    "cl_axi_sc_1x1_wrapper.v": "43dd040c56dfeb49d4b9e72ddee9e46be54bea9f381f5522af23907b8cdd091a",
}
VENDOR_PRODUCT_SUFFIXES = {".bd", ".v", ".sv", ".vhd", ".xdc", ".xci", ".dcp", ".mem", ".dat", ".coe"}
VENDOR_COPY_LIMIT = 1 << 20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_support():
    manifest = json.loads((SUPPORT / "manifest.json").read_text())
    for name, digest in manifest["sources"].items():
        if sha(SUPPORT / name) != digest:
            raise RuntimeError(f"Frozen F2 support changed: {name}")
    return manifest


def support_builder():
    # Its checked HDK, version and AFI-packaging helpers are reused unchanged.
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("c2_pinned_f2", SUPPORT / "boards/f2/build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Frozen support anchor changed: {old[:90]!r}")
    return text.replace(old, new, 1)


def smartconnect_products(hdk):
    """Inventory synthesis products only, not logs, simulation outputs or caches."""
    helper = support_builder()
    roots = [(hdk / helper.HBM_SC_BD).parent, (hdk / helper.HBM_SC_WRAPPER).parent.parent]
    products = {}
    for root in roots:
        if not root.is_dir():
            raise RuntimeError(f"Missing HDK SmartConnect product tree: {root}")
        for path in sorted(root.rglob("*")):
            # Path.rglob does not recurse into directory symlinks; refuse to
            # silently omit input trees. Ordinary file links are hashed/copied.
            if path.is_symlink() and path.is_dir():
                raise RuntimeError(f"Unreviewed directory link in SmartConnect products: {path}")
            if path.is_file() and path.suffix.lower() in VENDOR_PRODUCT_SUFFIXES:
                size = path.stat().st_size
                products[str(path.relative_to(hdk))] = {
                    "sha256": sha(path), "bytes": size,
                    "preserved": path.suffix.lower() != ".dcp" and size <= VENDOR_COPY_LIMIT,
                }
    return products


def check_smartconnect(hdk):
    """Check the generated HDK interface actually wired by the pinned C2 top.

    The vendor port name/metadata is aclk_250; it is driven by the actual 200 MHz
    core clock. Record the metadata unchanged so synthesis/CDC can audit it.
    """
    helper = support_builder()
    bd, wrapper = hdk / helper.HBM_SC_BD, hdk / helper.HBM_SC_WRAPPER
    interfaces = helper.check_hbm_sc(bd, wrapper)
    for name in ("AXI4", "AXI3"):
        if interfaces[name].get("ADDR_WIDTH") != "64":
            raise RuntimeError(f"{bd}: {name} address width must be 64")
    declarations = {}
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", wrapper.read_text(), flags=re.S)
    for direction, high, low, name in re.findall(
            r"\b(input|output)\s+(?:wire\s+)?(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*)?(\w+)\s*;",
            text):
        if name in declarations:
            raise RuntimeError(f"{wrapper}: duplicate port declaration {name}")
        declarations[name] = (direction, abs(int(high) - int(low)) + 1 if high else 1)
    expected = {"aclk_250": ("input", 1), "aclk_450": ("input", 1),
                "aresetn_250": ("input", 1)}
    # Only named connections in the top are required. AXI3 lock/cache/prot/qos
    # outputs are deliberately open; they still have their specified widths.
    for bus, data, ids, master in [("AXI4", 512, 16, False), ("AXI3", 256, 0, True)]:
        forward = {"awaddr": 64, "awlen": 8 if bus == "AXI4" else 4,
                   "awsize": 3, "awburst": 2, "awvalid": 1,
                   "wdata": data, "wstrb": data // 8, "wlast": 1, "wvalid": 1,
                   "bready": 1, "araddr": 64, "arlen": 8 if bus == "AXI4" else 4,
                   "arsize": 3, "arburst": 2, "arvalid": 1, "rready": 1}
        backward = {"awready": 1, "wready": 1, "bresp": 2, "bvalid": 1,
                    "arready": 1, "rdata": data, "rresp": 2, "rlast": 1, "rvalid": 1}
        if ids:
            forward.update(awid=ids, arid=ids)
            backward.update(bid=ids, rid=ids)
        for channel in ("aw", "ar"):
            forward.update({channel + "lock": 1 if bus == "AXI4" else 2,
                            channel + "cache": 4, channel + "prot": 3, channel + "qos": 4})
        for fields, direction in [(forward, "output" if master else "input"),
                                  (backward, "input" if master else "output")]:
            expected.update({bus + "_" + name: (direction, width) for name, width in fields.items()})
    for name, wanted in expected.items():
        if declarations.get(name) != wanted:
            raise RuntimeError(f"{wrapper}: {name} is {declarations.get(name)}, expected {wanted}")
    design = json.loads(bd.read_text())["design"]
    components = design.get("components", {})
    sc = components.get("smartconnect_0", {})
    values = {k: v.get("value") for k, v in sc.get("parameters", {}).items()}
    if sc.get("vlnv") != "xilinx.com:ip:smartconnect:1.0" or any(
            values.get(k) != v for k, v in {"NUM_CLKS": "2", "NUM_MI": "1", "NUM_SI": "1"}.items()):
        raise RuntimeError(f"{bd}: expected one 1x1 SmartConnect with two clocks")
    for path in (bd, wrapper):
        if sha(path) != SMARTCONNECT_SHA256[path.name]:
            raise RuntimeError(f"Unreviewed HDK SmartConnect bytes: {path}")
    return {"interfaces": interfaces, "ports": design.get("ports", {}), "components": components,
            "driven_clock_mhz": {"aclk_250": 200, "aclk_450": 450},
            "files": smartconnect_products(hdk)}


def check_vendor_unchanged(hdk, record):
    current = smartconnect_products(hdk)
    if current != record["files"]:
        changed = sorted(name for name in current.keys() | record["files"].keys()
                         if current.get(name) != record["files"].get(name))
        raise RuntimeError(f"HDK SmartConnect input changed during build: {changed[:8]}")


def check_clocks(top):
    """Reject illegal or frequency-changing MMCM recipes before freezing inputs.

    Attribute limits: AMD UG572 MMCM Attributes. Frequency limits: AMD DS923,
    MMCM Switching Characteristics, Virtex UltraScale+ -2 at 0.85 V.
    Fractions keep unsupported decimal values (notably 4.8) from rounding away.
    """
    for instance, input_mhz, outputs in [
        ("u_mmcm", 250, {"CLKOUT0_DIVIDE_F": 200, "CLKOUT1_DIVIDE": 400}),
        ("u_hmmcm", 100, {"CLKOUT0_DIVIDE_F": 450}),
    ]:
        blocks = [body for body, name in re.findall(
            r"MMCME4_ADV\s*#\((.*?)\)\s+(\w+)\s*\(", top, re.S) if name == instance]
        if len(blocks) != 1:
            raise RuntimeError(f"Expected one clock primitive: {instance}")
        values = {key: Fraction(value) for key, value in
                  re.findall(r"\.(\w+)\(([0-9.]+)\)", blocks[0])}

        def require(ok, reason):
            if not ok:
                raise RuntimeError(f"Invalid {instance} clock recipe: {reason}")

        require({"CLKIN1_PERIOD", "DIVCLK_DIVIDE", "CLKFBOUT_MULT_F", *outputs} <= values.keys(),
                "missing numeric attribute")
        divide, mult = values["DIVCLK_DIVIDE"], values["CLKFBOUT_MULT_F"]
        require(divide.denominator == 1 and 1 <= divide <= 106, "DIVCLK_DIVIDE")
        require(2 <= mult <= 128 and (mult * 8).denominator == 1, "CLKFBOUT_MULT_F granularity/range")
        require(values["CLKIN1_PERIOD"] == Fraction(1000, input_mhz), "input period")
        pfd = Fraction(input_mhz, 1) / divide
        vco = pfd * mult
        require(10 <= pfd <= 500 and 800 <= vco <= 1600, "PFD/VCO frequency")
        for name, mhz in outputs.items():
            output_divide = values[name]
            if name.endswith("_F"):
                legal = output_divide == 1 or (2 <= output_divide <= 128 and
                                               (output_divide * 8).denominator == 1)
            else:
                legal = 1 <= output_divide <= 128 and output_divide.denominator == 1
            require(legal, f"{name} granularity/range")
            require(vco / output_divide == mhz, f"{name} must produce {mhz} MHz")


def make_top():
    top = (SUPPORT / "boards/f2/design/cl_opentpu_hbm.sv").read_text()
    # The frozen source's M=4.8 is not a legal 1/8 multiplier. Keep M/D=4.8
    # with integer M=24, D=5: 50 MHz PFD, 1200 MHz VCO, 200/400 MHz outputs.
    top = replace_once(top,
        ".CLKIN1_PERIOD(4.0), .DIVCLK_DIVIDE(1), .CLKFBOUT_MULT_F(4.8)",
        ".CLKIN1_PERIOD(4.0), .DIVCLK_DIVIDE(5), .CLKFBOUT_MULT_F(24.0)")
    top = replace_once(top, "otpu_f2_hbm_sys #(", "otpu_f2_c2_sys #(")
    top = replace_once(top, "32'h4f545055; // OTPU", "32'h43324842; // C2HB")
    check_clocks(top)
    return top


def make_sys():
    text = (SUPPORT / "rtl/boards/f2/otpu_f2_hbm_sys.sv").read_text()
    start = text.index("module otpu_f2_hbm_sys #(")
    ports = text.index(") (", start)
    text = "// C2 kernel shell; generated from the pinned F2 HBM fabric.\n" + (
        "module otpu_f2_c2_sys #(parameter int S=1, NCH=8, CORE_KHZ=200000,\n"
        "  parameter logic [31:0] BUILD_ID=0\n" + text[ports:])
    text = replace_once(text, "NPORT > 32 || CH_LOG2 != 29", "NPORT != 8")
    text = replace_once(text, ".NPORT(NPORT), .CH_LOG2(CH_LOG2)", ".NPORT(NPORT), .CH_LOG2(29)")
    text = text.replace(
        "otpu_f2_hbm_sys: at most 32 HBM ports of 512 MiB (S x NCH <= 32, CH_LOG2 29)",
        "otpu_f2_c2_sys: exactly eight HBM ports required")
    text = text.replace("HBM ready, into the core clock for the calibration bits (otpu_board synchronizes it again)",
                        "HBM ready synchronized from the shell into the kernel clock")
    text = replace_once(text, "  logic hbm_rdy;\n  assign hbm_rdy = hbm_ready;", (
        '  (* ASYNC_REG = "TRUE" *) logic [1:0] ready_sync;\n'
        "  always_ff @(posedge clk)\n"
        "    if(rst) ready_sync <= 0; else ready_sync <= {ready_sync[0],hbm_ready};\n"
        "  wire hbm_rdy = ready_sync[1];"))
    start = text.index("  // ---- the accelerator (HAS_I2C")
    old = text[start:]
    connections = old[old.index("    .s_ctl_awaddr"):old.rindex("endmodule")]
    text = text[:start] + (
        "  otpu_c2_hbm_kernel #(.BUILD_ID(BUILD_ID), .CORE_KHZ(CORE_KHZ)) u_board (\n"
        "    .clk, .rst, .hbm_ready(hbm_rdy), .led,\n" + connections + "endmodule\n")
    return text


def configure_scripts(cl):
    scripts, constraints = cl / "build/scripts", cl / "build/constraints"
    synth = (scripts / "synth_cl_opentpu_hbm.tcl").read_text()
    synth = synth.replace("general.maxThreads 6", "general.maxThreads 4")
    synth = replace_once(synth, "read_xdc -unmanaged ${constraints_dir}/otpu_clk2x.tcl\n", "")
    start = synth.index("if {![llength $c2x_pb]")
    end = synth.index("report_utilization", start)
    synth = synth[:start] + synth[end:]
    (scripts / "synth_cl_opentpu.tcl").write_text(synth)
    impl = (scripts / "build_level_1_cl.tcl").read_text()
    impl = impl.replace("general.maxThreads 8", "general.maxThreads 4")
    impl = replace_once(impl, "  read_xdc -unmanaged ${::constraints_dir}/otpu_clk2x.tcl\n", "")
    (scripts / "build_level_1_cl.tcl").write_text(impl)

    timing = (constraints / "cl_timing_user_hbm.tcl").read_text()
    # Round 5 has only the host's three custom afifos and one gray write counter.
    # Vendor SmartConnect CDC constraints are audited separately after synthesis.
    timing = replace_once(timing,
        'foreach f $fifos {',
        'if {[llength $fifos] != 3} { error "C2 HBM CDC: expected exactly three host FIFOs" }\nforeach f $fifos {')
    timing = replace_once(timing,
        'if {[llength $counters] < 1} { error "openTPU CDC: expected the host path" }',
        'if {[llength $counters] != 1} { error "C2 HBM CDC: expected exactly one host write counter" }')
    # Tcl string-match bracket classes in the frozen R5 gate reject real names
    # such as g_sc[7].u_sc/.../src_gray_ff_reg[5]. Anchored regexps bound the
    # exact eight instances, wrapper hierarchy and three reset bits instead.
    timing = replace_once(timing,
        '  if {![string match {g_sc[[]*[]].u_sc/*} $n] && ![string match {u_hbm/*} $n] &&\n'
        '      ![string match {hbm_reset_n_reg*} $n]} { lappend otpu_on_hclk $n }',
        r'  if {![regexp {^g_sc\[[0-7]\]\.u_sc/.+$} $n] && ![regexp {^u_hbm/.+$} $n] &&' + '\n' +
        r'      ![regexp {^hbm_reset_n_reg\[[0-2]\]$} $n]} { lappend otpu_on_hclk $n }')
    (constraints / "cl_timing_user.tcl").write_text(timing)
    pnr = (constraints / "cl_pnr_user.tcl").read_text().replace(
        "foreach stem {main_reset_n bridge_reset_n core_reset_n}",
        "foreach stem {main_reset_n bridge_reset_n core_reset_n hbm_reset_n}")
    floorplan = (constraints / "cl_pnr_user_hbm.tcl").read_text()
    start = floorplan.index("########################################\n# pb_otpu_tmem:")
    end = floorplan.index("########################################\n# pb_otpu_fab:", start)
    floorplan = floorplan[:start] + floorplan[end:]
    floorplan = replace_once(floorplan,
        "foreach pb {pb_otpu_slr0 pb_otpu_tmem pb_otpu_fab pb_otpu_pcis pb_otpu_xs}",
        "foreach pb {pb_otpu_slr0 pb_otpu_fab pb_otpu_pcis pb_otpu_xs}")
    floorplan = replace_once(floorplan, "pblocks=5", "pblocks=4")
    (constraints / "cl_pnr_user.tcl").write_text(pnr + "\n" + floorplan)

    shutil.copy2(HERE / "validate_hbm_route.tcl", scripts / "validate_route.tcl")
    for name in ("classify_methodology.tcl", "check_scoped_clocks.tcl"):
        shutil.copy2(HERE / name, scripts / name)


def prepare(out, hdk, shell, build_id):
    if out.exists():
        raise RuntimeError(f"Refusing to reuse run directory: {out}")
    support = verify_support()
    smartconnect = check_smartconnect(hdk)
    for name in CORE_SOURCES:
        if not (ROOT / "rtl" / name).is_file():
            raise RuntimeError(f"Missing C2 kernel source: {name}")
    cl = out / "cl_opentpu"
    shutil.copytree(SUPPORT / "boards/f2/build", cl / "build")
    design = cl / "design"
    design.mkdir()
    (design / "cl_opentpu_hbm.sv").write_text(make_top())
    shutil.copy2(SUPPORT / "boards/f2/design/cl_id_defines.vh", design)
    shutil.copy2(hdk / "hdk/cl/examples/cl_mem_perf/design/cl_mem_hbm_wrapper.sv", design)
    (design / "opentpu_geometry.vh").write_text("localparam int S=1;\nlocalparam int NCH=8;\n")
    (design / "opentpu_params.vh").write_text(f".BUILD_ID(32'h{build_id}), .CORE_KHZ(200000)\n")
    files = []
    for parent, names in [(ROOT, CORE_SOURCES), (SUPPORT, FABRIC_SOURCES)]:
        for name in names:
            dest = design / "rtl" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parent / "rtl" / name, dest)
            files.append("rtl/" + name)
    (design / "otpu_f2_c2_sys.sv").write_text(make_sys())
    files.append("otpu_f2_c2_sys.sv")
    (design / "sources.tcl").write_text("set otpu_sources {\n" + "\n".join(files) + "\n}\n")
    configure_scripts(cl)
    (cl / "build/scripts/build_all.tcl").symlink_to(shell / "build/scripts/build_all.tcl")
    # Preserve small synthesis source products and hash every selected product,
    # including generated child-IP checkpoints. Vivado still reads installed
    # HDK paths; these are evidence copies, not relocated IP inputs.
    for name, record in smartconnect["files"].items():
        if not record["preserved"]:
            continue
        target = out / "vendor-inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hdk / name, target)
    check_vendor_unchanged(hdk, smartconnect)
    provenance = {
        "kind": "c2-hbm-kernel", "build_id": build_id, "core_khz": 200000,
        "part": PART, "hbm_ports": 8, "platform": {"mem": "hbm"},
        "support_commit": support["commit"], "support_manifest_sha256": sha(SUPPORT / "manifest.json"),
        "smartconnect": smartconnect, "implementation_tuning": TUNING,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "builder_sha256": sha(__file__),
        "sources": {str(p.relative_to(cl)): sha(p) for p in sorted(cl.rglob("*")) if p.is_file()},
    }
    (out / "inputs.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return cl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdk", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--flow", choices=["SynthCL", "BuildAll"], default="SynthCL")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--package", action="store_true")
    a = parser.parse_args()
    if a.package and a.flow != "BuildAll":
        parser.error("--package requires --flow BuildAll")
    verify_support()
    helper = support_builder()
    target = SimpleNamespace(platform=tomllib.loads((SUPPORT / "targets/f2-hbm.toml").read_text())["platform"])
    hdk, out = a.hdk.resolve(), a.out.resolve()
    shell = helper.check_hdk(hdk, target, shell=a.flow == "BuildAll")
    build_id = subprocess.check_output(["git", "rev-parse", "--short=8", "HEAD"], cwd=ROOT, text=True).strip()
    cl = prepare(out, hdk, shell, build_id)
    env = dict(os.environ, CL_DIR=str(cl), CL="cl_opentpu", SHELL_MODE="small_shell",
               BUILD_FLOW=a.flow, BUILD_TAG=build_id, ENCRYPT="0", HDK_SHELL_DIR=str(shell),
               HDK_SHELL_DESIGN_DIR=str(shell / "design"), HDK_COMMON_DIR=str(hdk / "hdk/common"),
               HDK_IP_SRC_DIR=str(hdk / "hdk/common/ip/cl_ip/cl_ip.srcs/sources_1/ip"),
               HDK_BD_SRC_DIR=str(hdk / "hdk/common/ip/cl_ip/cl_ip.srcs/sources_1/bd"),
               HDK_BD_GEN_DIR=str(hdk / "hdk/common/ip/cl_ip/cl_ip.gen/sources_1/bd"),
               VIVADO_TOOL_VERSION="2025.2", OTPU_RETIME="1", OTPU_FLOORPLAN="c2-four")
    cmd = ["vivado", "-mode", "batch", "-source", "build_all.tcl", "-log", str(out / "vivado.log"),
           "-journal", str(out / "vivado.jou"), "-tclargs", TUNING["place"], TUNING["phys"],
           TUNING["route"], "A1", "B2", "C0", "H2"]
    (out / "invocation.json").write_text(json.dumps({"command": cmd, "cwd": str(cl / "build/scripts"),
        "environment": {k: env[k] for k in env if k.startswith(("HDK_", "OTPU_")) or k in
                        ["CL_DIR", "CL", "SHELL_MODE", "BUILD_FLOW", "BUILD_TAG", "ENCRYPT", "VIVADO_TOOL_VERSION"]}}, indent=2) + "\n")
    if a.prepare_only:
        print(f"Prepared {cl}")
        return
    helper.check_vivado_version(helper.command("vivado", "-version"), "2025.2")
    check_vendor_unchanged(hdk, json.loads((out / "inputs.json").read_text())["smartconnect"])
    with (out / "console.log").open("w") as log:
        result = subprocess.run(cmd, cwd=cl / "build/scripts", env=env, stdout=log, stderr=subprocess.STDOUT)
    (out / "exit_code").write_text(str(result.returncode) + "\n")
    check_vendor_unchanged(hdk, json.loads((out / "inputs.json").read_text())["smartconnect"])
    stage = "post_synth" if a.flow == "SynthCL" else "post_route"
    dcp = cl / f"build/checkpoints/cl_opentpu.{build_id}.{stage}.dcp"
    if result.returncode or not dcp.is_file():
        raise RuntimeError(f"{a.flow} failed: {out / 'console.log'}")
    if a.flow == "BuildAll" and not dcp.with_name(f"cl_opentpu.{build_id}.route_validation.passed").is_file():
        raise RuntimeError("Route qualification failed; refusing to package")
    artifacts = [dcp]
    if a.package:
        artifacts.append(helper.package_afi(cl, shell, hdk, build_id, env))
    (out / "outputs.json").write_text(json.dumps({"flow": a.flow, "inputs_sha256": sha(out / "inputs.json"),
        "artifacts": {str(p.relative_to(out)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in artifacts}}, indent=2) + "\n")
    print(f"{a.flow} complete: {dcp}")


if __name__ == "__main__":
    main()
