// AMD HBM PG276 requires 32-byte alignment and AxSIZE=5. The AWS HBM
// wrapper already fixes AxSIZE=5. SmartConnect packs every multi-beat
// transaction, but keeps single-beat narrow SIZE/address values (PG247).
// Align the address while preserving WSTRB and all byte-lane positions.
// Reading the surrounding 32-byte word is safe for HBM ordinary memory.
module otpu_hbm_geometry (
 input wire [33:0] s_awaddr,s_araddr,
 output wire [33:0] m_awaddr,m_araddr
);
 assign m_awaddr={s_awaddr[33:5],5'b0};
 assign m_araddr={s_araddr[33:5],5'b0};
endmodule
