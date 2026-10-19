"""C2 Alex-derived wiring and IP topology; no vendor or FPGA timing claims."""
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("c2_alex_generate", ROOT / "rtl/c2_alex/generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hdk(tmp_path):
    directory = tmp_path / "hdk"
    interfaces = directory / "common/shell_stable/design/interfaces"
    interfaces.mkdir(parents=True)
    (interfaces / "cl_ports.vh").write_text("\n".join([
        "input logic clk_main_a0, clk_hbm_ref, rst_main_n,",
        *[f"output logic [31:0] {p}," for p in
          ("cl_sh_id0", "cl_sh_id1", "cl_sh_status0", "cl_sh_status1", "cl_sh_status2")],
        "output logic [15:0] cl_sh_status_vled,",
        "output logic unused_shell_output,",
        "output logic template_driven_output",
    ]))
    for name in ("unused_ddr_template.inc", "unused_cl_sda_template.inc",
                 "unused_flr_template.inc", "unused_apppf_irq_template.inc"):
        (interfaces / name).write_text("// offline shell header fixture\n")
    (interfaces / "unused_flr_template.inc").write_text("assign template_driven_output = 1'b1;\n")
    return directory


def test_top_retains_eight_native_channels_and_kernel_abi(generator, hdk):
    top = generator.render_top(hdk, "4edf7e18")
    assert top == generator.render_top(hdk, "4EDF7E18")
    assert len(re.findall(r"\botpu_c2_hbm_kernel #", top)) == 1
    assert "otpu_board" not in top
    assert ".RD(32)" in top and ".DEPTH(256)" in top and ".MAX_ROWS(512)" in top
    assert ".BUILD_ID(32'h4edf7e18)" in top and ".CORE_KHZ(200000)" in top
    assert ".NUM_OF_AXI_PORTS(8)" in top
    assert "assign cl_sh_status0 = 32'h43324842;" in top
    assert "assign cl_sh_status1 = 32'h4edf7e18;" in top
    assert "assign unused_shell_output = '0;" in top
    assert "assign template_driven_output" not in top  # header template owns it
    assert "pcis_wbusy" not in top and "otpu_axil_cdc" not in top
    assert ".m_axi_aresetn(core_rst_n)" in top
    assert ".hbm_ready(hbm_ready_core)" in top and ".rst(core_rst)" in top
    assert "always_ff @(posedge core_clk) core_rst <= !core_rst_n;" in top
    assert "@(posedge hbm_clk)" not in top
    assert ".WIDTH(11)" in top and ".src_in({n_err,leds})" in top
    assert "clk_f" not in top  # C2 has no pumped-TMEM client or multicycle
    for p in range(8):
        instance = re.search(r"otpu_native_axi #\((.*?)\) u_native" + str(p) + r" \((.*?)\n\);", top, re.S)
        assert instance is not None and ".BASE_ADDR(34'd0)" in instance[1]
        for pin in ("cvalid", "cready", "cwe", "caddr", "wvalid", "wready", "wdata",
                    "wmask", "rvalid", "rdata", "wdone", "err"):
            assert f".n_{pin}(n_{pin}[{p}])" in instance[2]
        assert f".S_N{p}_araddr(n{p}_araddr)" in top
        assert f".S_N{p}_arlen(n{p}_arlen)" in top
        assert f".S_N{p}_awlen(8'd0)" in top
        assert f".S_N{p}_arcache(4'b0000)" in top
        assert f".M_H{p}_rdata(h_rdata[{p}])" in top
        assert f".M_H{p}_awaddr(h_awaddr[{p}])" in top
        assert f".m_awaddr(hb_awaddr[{p}])" in top
        assert f"assign h_awid[{p}] = 6'd0;" in top
        assert f"n_caddr[{p}][24:23]!=0" in top
    assert "u_native8" not in top and "M_H8_" not in top


@pytest.mark.parametrize("bad", ["", "123", "123456789", "xyz56789", "12;34567"])
def test_build_identity_is_validated(generator, hdk, bad):
    with pytest.raises(ValueError, match="eight hexadecimal"):
        generator.render_top(hdk, bad)


def test_cli_emits_deterministic_top_and_ip(generator, hdk, tmp_path):
    top, ip = tmp_path / "out/top.sv", tmp_path / "out/ip.tcl"
    result = subprocess.run([sys.executable, str(ROOT / "rtl/c2_alex/generate.py"),
                             "--hdk", str(hdk), "--output", str(top), "--build-id", "0123abcd",
                             "--ip-script", str(ip)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert top.read_text() == generator.render_top(hdk, "0123abcd")
    assert ip.read_text() == generator.render_memory_ip()


TCL_SHIM = r'''
proc current_project {} {return offline_project}
proc set_param {args} {puts "PARAM\t[join $args \t]"}
proc create_project {args} {}
proc import_ip {args} {}
proc upgrade_ip {args} {}
proc report_property {args} {}
proc create_ip {args} {}
proc create_bd_design {args} {}
proc create_bd_port {args} {return [lindex $args end]}
proc create_bd_cell {args} {return [lindex $args end]}
proc create_bd_intf_port {args} {
  puts "PORT\t[lindex $args end]\t[lindex $args 1]"
  return [lindex $args end]
}
proc set_property {args} {
  if {[lindex $args 0] eq "-dict"} {
    foreach {k v} [lindex $args 1] {puts "PROP\t[lindex $args 2]\t$k\t$v"}
  } else {puts "PROP\t[lindex $args 2]\t[lindex $args 0]\t[lindex $args 1]"}
}
proc get_property {key object} {
  if {$key eq "GENERATE_SYNTH_CHECKPOINT"} {return 0}
  if {$key eq "PROGRESS"} {return "100%"}
  return "$object.xci"
}
foreach command {get_ips get_bd_ports get_bd_pins get_bd_intf_pins get_bd_addr_spaces get_bd_addr_segs get_files} {
  proc $command {args} {return [lindex $args end]}
}
proc connect_bd_net {args} {puts "CLOCK\t[join $args \t]"}
proc connect_bd_intf_net {args} {puts "CONNECT\t[join $args \t]"}
proc assign_bd_address {args} {
  puts "MAP\t[lindex $args 5]\t[lindex $args 6]\t[lindex $args 1]\t[lindex $args 3]"
}
proc exclude_bd_addr_seg {args} {puts "EXCLUDE\t[lindex $args 1]\t[lindex $args 2]"}
foreach command {validate_bd_design save_bd_design generate_target make_wrapper write_bd_tcl create_ip_run wait_on_run} {
  proc $command {args} {}
}
proc get_runs {args} {return offline_run}
proc launch_runs {args} {puts "JOBS\t[lindex $args end]"}
'''


def execute_recipe(generator, tmp_path, jobs="1"):
    tclsh = shutil.which("tclsh")
    if not tclsh:
        pytest.skip("tclsh required for topology regression")
    driver = tmp_path / "driver.tcl"
    driver.write_text(TCL_SHIM + "\n" + generator.render_memory_ip())
    return subprocess.run([tclsh, str(driver), str(tmp_path), str(tmp_path / "hdk"), jobs],
                          capture_output=True, text=True)


def test_actual_tcl_builds_exact_routing_matrix(generator, tmp_path):
    result = execute_recipe(generator, tmp_path, "2")
    assert result.returncode == 0, result.stderr
    rows = [line.split("\t") for line in result.stdout.splitlines()]
    props = {(row[1], row[2]): row[3] for row in rows if row[0] == "PROP"}
    assert props["memory_sc", "CONFIG.NUM_SI"] == "9"
    assert props["memory_sc", "CONFIG.NUM_MI"] == "8"
    assert props["memory_sc", "CONFIG.NUM_CLKS"] == "3"
    assert {tuple(row[1:]) for row in rows if row[0] == "CLOCK"} == {
        ("core_clk", "memory_sc/aclk"), ("host_clk", "memory_sc/aclk1"),
        ("hbm_clk", "memory_sc/aclk2"), ("aresetn", "memory_sc/aresetn")}
    assert {tuple(row[1:]) for row in rows if row[0] == "CONNECT"} == {
        ("S_HOST", "memory_sc/S00_AXI"),
        *[(f"S_N{p}", f"memory_sc/S{p+1:02d}_AXI") for p in range(8)],
        *[(f"M_H{p}", f"memory_sc/M{p:02d}_AXI") for p in range(8)]}
    assert len([r for r in rows if r[0] == "PORT"]) == 17
    for p in range(8):
        assert props[f"S_N{p}", "CONFIG.DATA_WIDTH"] == "512"
        assert props[f"S_N{p}", "CONFIG.ID_WIDTH"] == "1"
        assert props[f"M_H{p}", "CONFIG.PROTOCOL"] == "AXI3"
        assert props[f"M_H{p}", "CONFIG.DATA_WIDTH"] == "256"
    assert props["S_HOST", "CONFIG.ADDR_WIDTH"] == "64"
    assert props["S_HOST", "CONFIG.ID_WIDTH"] == "16"
    assert props["core_clk", "CONFIG.FREQ_HZ"] == "200000000"
    assert props["host_clk", "CONFIG.FREQ_HZ"] == "250000000"
    assert props["hbm_clk", "CONFIG.FREQ_HZ"] == "450000000"
    maps = [(r[1], r[2], int(r[3], 0), int(r[4], 0)) for r in rows if r[0] == "MAP"]
    assert len(maps) == 16
    assert set(maps) == {(src, f"M_H{p}/Reg", p << 29, 1 << 29)
                         for p in range(8) for src in ("S_HOST", f"S_N{p}")}
    excluded = [tuple(r[1:]) for r in rows if r[0] == "EXCLUDE"]
    assert len(excluded) == 56
    assert set(excluded) == {(f"S_N{p}", f"M_H{q}/Reg") for p in range(8) for q in range(8) if p != q}
    assert [r for r in rows if r[0] == "JOBS"] == [["JOBS", "2"]]
    assert [r for r in rows if r[0] == "PARAM"] == [["PARAM", "general.maxThreads", "4"]]
    contract = generator.memory_contract()
    assert contract["total_bytes"] == 1 << 32 and contract["kernel_receive_depth"] == 32
    assert contract["native_max_outstanding"] == 128
    assert [p["base"] for p in contract["native"]] == [p << 29 for p in range(8)]


@pytest.mark.parametrize("jobs", ["0", "4", "-1", "1.5", "bad"])
def test_ip_job_bound_is_enforced(generator, tmp_path, jobs):
    result = execute_recipe(generator, tmp_path, jobs)
    assert result.returncode != 0 and "C2 IP jobs must be 1..3" in result.stderr
    assert not (tmp_path / "ip_done.txt").exists()


def test_wrapper_mapping_matches_existing_host_layout(generator, hdk):
    from opentpu.c2_f2 import make_fixture
    top = generator.render_top(hdk, "01020304")
    bases = {int(p): int(base, 16) for p, base in re.findall(
        r"assign n(\d)_araddr = n\d_araddr_local \| 34'h([0-9a-f]+);", top)}
    assert len(bases) == 8
    for rows, kb in [(1, 1), (3, 9), (9, 33), (512, 255)]:
        groups = (rows + 3) // 4
        # Place each fixture at the exact last legal allocation in its PC.
        fixture = make_fixture(rows, kb, seed=3, base_beat=(1 << 23) - groups * kb)
        banks = fixture.banks()
        for r in range(rows):
            for half in range(2):
                p = 2 * (r % 4) + half
                for k in {0, kb-1}:
                    beat = fixture.base_beat + (r // 4) * kb + k
                    address = bases[p] | (beat << 6)
                    assert address == (p << 29) + beat * 64
                    assert address >> 29 == p and address + 64 <= (p+1) << 29
                    offset = ((r // 4) * kb + k) * 64
                    expected = fixture.weights[r, k, half*64:(half+1)*64].view("uint8")
                    assert bytes(banks[p][offset:offset+64]) == bytes(expected)


def test_generated_native_wiring_runs_frozen_adapter_at_all_pc_boundaries(generator, hdk, tmp_path):
    compiler, runtime = shutil.which("iverilog"), shutil.which("vvp")
    if not compiler or not runtime:
        pytest.skip("Icarus Verilog is required for native wiring simulation")
    top = generator.render_top(hdk, "11223344")
    # Compile the exact generated eight native instances and address transforms;
    # a small independent bus driver replaces kernel/vendor IP, not the adapter.
    native = top[top.index("logic [33:0] n0_awaddr;"):top.index("logic [33:0] h_araddr [0:7];")]
    decls = "\n".join("logic [7:0]" + (f"[{w-1}:0]" if w > 1 else "") + " n_" + n + ";"
        for n, w in {"cvalid":1,"cready":1,"cwe":1,"caddr":25,"wvalid":1,"wready":1,
                     "wdata":512,"wmask":64,"rvalid":1,"rdata":512,"wdone":16,"err":1}.items())
    ties, checks = [], []
    for p in range(8):
        ties += [f"assign n{p}_{n}={v};" for n, v in {
            "awready":"1'b1", "wready":"1'b1", "bvalid":"1'b0", "bresp":"2'b0",
            "arready":f"cycle%5!={p%5}", "rvalid":"1'b0", "rdata":"'0", "rlast":"1'b0", "rresp":"2'b0"}.items()]
        checks.append(f'''if(n{p}_arvalid&&n{p}_arready)begin
          if(n{p}_araddr[11:6]+int'(n{p}_arlen)>63)$fatal(1,"crossed page");
          for(int b=0;b<=n{p}_arlen;b++)begin
            if(n{p}_araddr+64*b != 34'h{p<<29:09x}+64*beats[seen[{p}]])$fatal(1,"wrong PC/address");
            seen[{p}]++;
          end
        end''')
    bench = tmp_path / "native.sv"
    bench.write_text('''module tb;
      logic core_clk=0,core_rst=1; always #5 core_clk=!core_clk;
      integer cycle=0; integer seen[8]; integer beats[6];
    ''' + decls + "\n" + native + "\n" + "\n".join(ties) + '''
      always @(posedge core_clk)begin cycle<=cycle+1;if(!core_rst)begin
    ''' + "\n".join(checks) + '''
      end end
      initial begin
        for(int p=0;p<8;p++)seen[p]=0;
        beats[0]=0;beats[1]=1;beats[2]=62;beats[3]=63;beats[4]=64;beats[5]='h7fffff;
        if($test$plusargs("invalid"))beats[5]='h800000;
        n_cvalid=0;n_cwe=0;n_wvalid=0;n_wdata=0;n_wmask=0;n_caddr=0;
        repeat(4)@(negedge core_clk);core_rst=0;
        for(int b=0;b<6;b++)begin
          @(negedge core_clk);for(int p=0;p<8;p++)n_caddr[p]=beats[b];n_cvalid='1;
          @(posedge core_clk);while(n_cready!='1)@(posedge core_clk);
          @(negedge core_clk);n_cvalid=0;
        end
        repeat(100)@(negedge core_clk);
        for(int p=0;p<8;p++)if(seen[p]!=6)$fatal(1,"missing reads");
        $display("ALL_EIGHT_PC_BOUNDARIES_PASS");$finish;
      end
    endmodule
    ''')
    binary = tmp_path / "native.vvp"
    adapter = ROOT / "boards/c2_f2_alex/support/rtl/boards/f2/otpu_native_axi.sv"
    result = subprocess.run([compiler, "-g2012", "-s", "tb", "-o", str(binary), str(adapter), str(bench)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([runtime, str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and "ALL_EIGHT_PC_BOUNDARIES_PASS" in result.stdout, result.stdout + result.stderr
    bad = subprocess.run([runtime, str(binary), "+invalid"], capture_output=True, text=True, timeout=30)
    assert bad.returncode != 0 and "exceeds 512 MiB window" in bad.stdout
