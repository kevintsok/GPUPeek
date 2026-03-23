# CUTLASS Tutorial: GEMM with Thread Block Clusters on NVIDIA Blackwell GPUs

Source: https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/

## Key Concepts

**Thread Block Clusters** group SMs that are physically close on-chip. Thread blocks in a cluster are guaranteed to be co-scheduled on SMs within the same GPU Processing Cluster (GPC). This feature, introduced in Hopper, enables distributed shared memory and cooperative loading across CTAs.

### Key Properties

- Maximum portable cluster size: 8
- Hopper H100 and Blackwell B200 support up to 16 with opt-in
- Cluster shape defined as `<cluster.x, cluster.y, cluster.z>`
- Shape must evenly divide the grid size

**TMA Multicast** loads the same tensor tile to multiple CTAs simultaneously. "Each CTA loads a portion of the data that is multicast into the SMEM of the other participating CTAs. For example, if the number of participating CTAs is 4, each CTA loads a quarter of the data."

## Implementation

### Launching Clusters in CUTLASS

```cpp
auto params = {dimGrid, dimBlock, dimCluster, smemBytes};
auto status = cutlass::launch_kernel_on_cluster(params, kernel_ptr, ...);
```

### TMA Multicast Atom

```cpp
Copy_Atom tma_atom_A = make_tma_atom(
    SM90_TMA_LOAD_MULTICAST{},
    mA, sA_layout,
    select<0,2>(mma_tiler),
    size<2>(cluster_layout_vmnk)
);
```

### Bitmask Generation

```cpp
int cta_rank = cute::block_rank_in_cluster();
uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, coord);
uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, coord);
```

### Synchronization Requirements

- CTA participation encoded via 16-bit `ctaMask` bitmask
- Each bit corresponds to a CTA in cluster
- CTAs wait on barriers for all participating TMA operations
- Must synchronize before MMA and before buffer overwrite

The article references CuTe Blackwell examples 3 and 4 from the NVIDIA CUTLASS repository for complete implementations.
