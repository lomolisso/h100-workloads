# h100-workloads

A catalogue of workloads for a project on 3D stacking of Hopper-like GPGPUs. Folding GPCs in 3D, further partitioning the
L2 and folding its partitions all change SMEM, DSMEM and L2 latencies and bandwidths. These workloads will be used to:

1. profile each workload on one H100 and on a [Vortex](https://github.com/vortexgpgpu/vortex) GPGPU modified to resemble
   it, to show that the modified Vortex is a good baseline;
2. run a sensitivity analysis on the simulator's latencies and bandwidths, to decide what to stack and which metric
   matters most.

**Status:** catalogue only. Nothing here has been profiled, and there are no run scripts yet. The list is still growing:
[`research_prompt.md`](research_prompt.md) briefs a research agent to find more workloads.

## Taxonomy

| Class | Name | Definition |
|---|---|---|
| **1a** | Cluster collectives | Use DSMEM and the SM-to-SM network for rooted (e.g. TMA multicast) or symmetric (e.g. ClusterReduce) collectives |
| **1b** | Shared memory | Use a thread-block cluster as a bigger block with more SMEM: point-to-point traffic, with no collective or only a trivial scalar one |
| **2a** | Benchmarks | No DSMEM; from an accepted suite such as Rodinia or the Accel-Sim workload set |
| **2b** | Scientific computing | No DSMEM; simulations and other codes used by physicists, mathematicians, biologists and others |
| **2c** | ML kernels | No DSMEM; machine-learning kernels such as tensor-core GEMM or attention |

## Workloads

Only workloads we ran ourselves on an H100 are listed. Each one is a single kernel, not a whole application.

| Class | Workload | Paper | Status | Repo |
|---|---|---|---|---|
| 1a | [`clusterfusion_llama2_7b_decoder`](workloads/1a_collectives/clusterfusion_llama2_7b_decoder) | [ClusterFusion](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3760d0ea4709a913a4804f4b4c073836-Abstract-Conference.html) | NeurIPS 2025 | [ClusterFusion](https://github.com/xinhao-luo/ClusterFusion) |
| 1a | [`clusterfusion_deepseek_v2_lite_mla`](workloads/1a_collectives/clusterfusion_deepseek_v2_lite_mla) | same | NeurIPS 2025 | same |
| 1a | [`clusterfusion_rmsnorm`](workloads/1a_collectives/clusterfusion_rmsnorm) | same | NeurIPS 2025 | same |
| 1a | [`dak_gemm_multicast`](workloads/1a_collectives/dak_gemm_multicast) | [DAK](https://doi.org/10.48550/arXiv.2604.26074) | arXiv preprint | [DirectAccessKernel](https://github.com/shouxulin/DirectAccessKernel) |
| 1a | [`deepgemm_fp8_gemm`](workloads/1a_collectives/deepgemm_fp8_gemm) | none | no paper | [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) |
| 1a | [`hpcops_topk_cluster8`](workloads/1a_collectives/hpcops_topk_cluster8) | [Sample-Guided Exact Top-K](https://arxiv.org/abs/2609.08450) | arXiv preprint | [hpc-ops](https://github.com/Tencent/hpc-ops) |
| 1b | [`credit_layernorm_backward`](workloads/1b_shared_memory/credit_layernorm_backward) | [CREDIT](https://arxiv.org/abs/2609.01864) | arXiv preprint | [CREDIT](https://github.com/zhengxiongli08/CREDIT) |
| 1b | [`credit_weighted_var_backward`](workloads/1b_shared_memory/credit_weighted_var_backward) | same | arXiv preprint | same |
| 1b | [`credit_pearson_backward`](workloads/1b_shared_memory/credit_pearson_backward) | same | arXiv preprint | same |
| 1b | [`credit_softmax_logits_backward`](workloads/1b_shared_memory/credit_softmax_logits_backward) | same | arXiv preprint | same |
| 1b | [`credit_lars_momentum`](workloads/1b_shared_memory/credit_lars_momentum) | same | arXiv preprint | same |
| 1b | [`credit_rowwise_quant`](workloads/1b_shared_memory/credit_rowwise_quant) | same | arXiv preprint | same |

CREDIT sits in 1b because its cluster mainly pools shared memory. It also runs an AllReduce of 1–5 scalars.

## Known risk: getting CUDA workloads onto Vortex

Vortex's README lists OpenCL 1.2, Vulkan and HIP as its software stack, with chipStar as a dependency. It does not list
CUDA. All 12 workloads above are CUDA kernels built around thread-block clusters, and most run through a Python or torch
host. In addition, ClusterFusion's `dsm.cuh`, DAK and top-k contain inline asm, and DAK and DeepGEMM use `wgmma`. Running
them on Vortex is unverified work, and it is separate from the problem of scaling them down.

## Layout

```
third_party/   the workloads' repositories as pinned submodules, never edited
workloads/
  <class>/<workload>/
    upstream        symlink to the workload's submodule; read the workload's code through this
    workload.yaml   what it does with DSMEM, source paths, entry kernel, paper, shapes run on the H100
    patches/        correctness fixes only, where the kernel is wrong without them
```

## Getting started

```bash
git clone git@github.com:lomolisso/h100-workloads.git
cd h100-workloads
git submodule update --init
git -C third_party/DeepGEMM submodule update --init   # DeepGEMM's cutlass and deep_jit, needed only to build it
```

## Adding a workload

1. Add its repository under `third_party/` as a submodule, pinned to a specific commit. Never edit a submodule.
2. Create `workloads/<class>/<name>/` with an `upstream` symlink and a `workload.yaml`.
3. Keep only the workload itself, without baselines to compare it against.
4. Add a patch only if the kernel is incorrect without it, and say what it fixes.
5. List the workload only once we have run it ourselves on an H100.
