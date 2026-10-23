// Internal typed local-controller contract. NOT a binary ISA/DMA descriptor ABI.
// The Python W4A16Job adapter resolves symbolic strings into required predicates.
package otpu_nvfp4_job;
  typedef struct packed {
    logic [63:0] address,span,stride;
  } buffer_t;
  typedef struct packed {
    logic [31:0] abi_major,abi_minor,flags,reserved;
    // launch, BF16 A, E2M1/K16/E4M3FN W, FP32 Y, multiply, bypass, post NONE
    logic [6:0] contract;
    logic [1:0] mode;
    logic [255:0] digest;
    logic [31:0] tokens,outputs,k,global_bits;
    buffer_t a,w,scales,y;
    logic [63:0] context_id,generation,request_id,position,epoch;
    logic [31:0] dependency_count;
  } descriptor_t;
  localparam logic [255:0] GRADUAL_DIGEST=256'h9c94b192649df89f5c37c851e10936526b26ddd8d7272145759ace26df45d6be;
  localparam logic [255:0] FTZ_DIGEST=256'hfedd4750edbc4b47c84f90a414090603343c7a06124b2348612216df2881b0dc;

  function automatic logic buffer_ok(input buffer_t b,input logic [63:0] bytes_needed,
                                      input integer alignment);
    logic [64:0] end_address;
    end_address={1'b0,b.address}+{1'b0,b.span};
    return b.span>=bytes_needed && b.stride>=bytes_needed &&
      end_address<=65'h10000000000000000 &&
      ((alignment==1) || (alignment==2 && !b.address[0] && !b.stride[0]) ||
       (alignment==4 && b.address[1:0]==0 && b.stride[1:0]==0));
  endfunction

  function automatic logic overlaps(input buffer_t a,input buffer_t b);
    logic [64:0] ae,be;
    ae={1'b0,a.address}+{1'b0,a.span};
    be={1'b0,b.address}+{1'b0,b.span};
    return {1'b0,a.address}<be && {1'b0,b.address}<ae;
  endfunction

  function automatic logic [5:0] validate(input descriptor_t d,
                                          input logic [63:0] session_generation);
    logic [5:0] faults;
    logic [63:0] k;
    k={32'd0,d.k};faults=0;
    faults[0]=d.abi_major!=1 || d.abi_minor!=0 || d.flags!=0 || d.reserved!=0 ||
              d.contract!=7'h7f || d.global_bits[31] || d.global_bits[30:23]==255;
    faults[1]=(d.mode==0 && d.digest!=GRADUAL_DIGEST) ||
              (d.mode==1 && d.digest!=FTZ_DIGEST) || d.mode>1;
    faults[2]=d.tokens!=1 || d.outputs!=1 || d.k==0 || d.k>64;
    faults[3]=!buffer_ok(d.a,k<<1,2) || !buffer_ok(d.w,(k+1)>>1,1) ||
              !buffer_ok(d.scales,(k+15)>>4,1) || !buffer_ok(d.y,64'd4,4) ||
              overlaps(d.y,d.a) || overlaps(d.y,d.w) || overlaps(d.y,d.scales);
    faults[4]=d.generation!=session_generation;
    faults[5]=d.dependency_count!=0;
    return faults;
  endfunction
endpackage
