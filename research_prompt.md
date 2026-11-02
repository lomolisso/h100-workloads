# Research brief: find GPGPU workloads for a 3D-stacked GPU study

You have no prior context on this project, so everything you need is here. Your job is to find more workloads in the
categories below and report them in the format of section 7. Do not run anything.

## 1. Project context

We are studying **3D stacking of Hopper-like GPGPUs** (NVIDIA H100 class). Stacking applies at the compute and memory
hierarchy levels: folding GPCs in 3D, further partitioning the L2, folding L2 partitions in 3D, and so on. In effect, this
changes the access latencies and bandwidths of three things:
- **SMEM**: the shared memory inside each SM.
- **DSMEM**: distributed shared memory, where the CTAs of a thread-block cluster access each other's SMEM over the
  SM-to-SM network.
- **L2**.

The plan for the workloads you find:
1. Run and profile each one individually on a real H100.
2. Run and profile the same workloads on a [Vortex GPGPU](https://github.com/vortexgpgpu/vortex) whose architecture we are
   modifying to resemble an H100, to show that the two correlate.
3. Change the simulator's latencies and bandwidths and measure each workload's sensitivity. This tells us what is worth
   stacking and which metric matters most.

Different workloads use different parts of the GPU, so **diversity of memory behaviour matters more than count**. We want
workloads that together stress SMEM, DSMEM, L2 and HBM in different ways.

The hardware is one **H100 NVL**: 132 SMs in 8 GPCs, about 228 KiB of SMEM per SM, 60 MiB of L2, 94 GB of HBM3, CUDA 12.9
toolchain.

## 2. Taxonomy

Give every workload exactly one class.

| Class | Name | Definition |
|---|---|---|
| **1a** | Cluster collectives | Uses DSMEM and the SM-to-SM network for **rooted** (e.g. TMA multicast) or **symmetric** (e.g. ClusterReduce, AllReduce) collective operations |
| **1b** | Shared memory | Uses a thread-block cluster as a bigger block with more SMEM. Traffic is point-to-point, with **no collective, or only a trivial scalar one** |
| **2a** | Benchmarks | Does not use DSMEM. Comes from a benchmark suite accepted in academia |
| **2b** | Scientific computing | Does not use DSMEM. Simulations and other codes that physicists, mathematicians, biologists and others run on GPGPUs |
| **2c** | ML kernels | Does not use DSMEM. Machine-learning kernels such as tensor-core GEMM or attention |

Rules for overlaps:
- Any real use of clusters and DSMEM makes it class 1, whatever its domain.
- Otherwise, use 2a if its main identity is being part of a benchmark suite, even if the problem is scientific.
- Use 2b if it is primarily a code that scientists use, and 2c if it is primarily an ML kernel.

## 3. What we already have

We ran these 12 kernels ourselves on the H100. **Do not report them again.** Other kernels from the same repositories are
welcome if they are distinct workloads.

| Class | Workload | Paper title | Paper URL | Repo | Publication status |
|---|---|---|---|---|---|
| 1a | Llama-2-7B fused decoder | ClusterFusion: Expanding Operator Fusion Scope for LLM Inference via Cluster-Level Collective Primitive | https://proceedings.neurips.cc/paper_files/paper/2025/hash/3760d0ea4709a913a4804f4b4c073836-Abstract-Conference.html | https://github.com/xinhao-luo/ClusterFusion | NeurIPS 2025 |
| 1a | DeepSeek-V2-Lite fused MLA decoder | same as above | same | same | NeurIPS 2025 |
| 1a | RMSNorm | same as above | same | same | NeurIPS 2025 |
| 1a | Multicast GEMM | DAK: Direct-Access-Enabled GPU Memory Offloading with Optimal Efficiency for LLM Inference | https://doi.org/10.48550/arXiv.2604.26074 | https://github.com/shouxulin/DirectAccessKernel | arXiv preprint |
| 1a | FP8 GEMM with TMA multicast | none (library) | none | https://github.com/deepseek-ai/DeepGEMM | no paper |
| 1a | Cluster-8 exact top-k | Sample-Guided Exact Top-K Selection for Long-Context Sparse Attention | https://arxiv.org/abs/2609.08450 | https://github.com/Tencent/hpc-ops | arXiv preprint |
| 1b | LayerNorm backward | CREDIT: Cost-guided Reduction-reuse with Efficient DSMEM Inter-CTA Tiling | https://arxiv.org/abs/2609.01864 | https://github.com/zhengxiongli08/CREDIT | arXiv preprint |
| 1b | Weighted variance backward | same as above | same | same | arXiv preprint |
| 1b | Pearson backward | same as above | same | same | arXiv preprint |
| 1b | Softmax logits backward | same as above | same | same | arXiv preprint |
| 1b | LARS momentum | same as above | same | same | arXiv preprint |
| 1b | Row-wise quantisation | same as above | same | same | arXiv preprint |

CREDIT is in 1b because its cluster mainly pools SMEM, although it also runs an AllReduce of 1–5 scalars. Its publication
status and DeepGEMM's are below the bar in section 5. They are listed because we already ran them, and they do not lower
the bar for new candidates.

**Already examined and excluded. Report these only if you find what is missing:**

| Work | Why excluded | What would change that |
|---|---|---|
| FlashAttention-4 (arXiv 2603.05451) | Its 2-CTA kernels are Blackwell-only | A version that runs on Hopper |
| hpc-ops Lamport RMSNorm | Cannot be measured on one GPU | A single-GPU mode that keeps the algorithm |
| FlashFuser (arXiv 2512.12949) | No public artifact found | A public artifact |
| ExpertPlex (arXiv 2607.18002) | No public code for its own Adaptive Persistent Kernel; its stated base, DeepGEMM, is not that kernel | A public artifact of that kernel |

## 4. Your task

Find more workloads in **all five classes**. Priorities:
- **1b needs the most attention.** Apart from CREDIT, which is not a pure example, we have no workload that uses a cluster
  as a bigger block. Look for stencils, halo exchange, large shared buffers, or other point-to-point DSMEM uses. One lead
  to check: Shan & Araya-Polo, "Evaluation of Programming Models and Performance for Stencil Computation on Current GPU
  Architectures" (arXiv 2404.04441). We have not checked whether it has an artifact.
- **1a:** new uses of rooted or symmetric collectives over DSMEM, beyond those above.
- **2a, 2b, 2c:** these are empty. Cover a wide spread of behaviour, not many variants of one pattern.

If a class is thin, say so. Do not pad it with weak candidates.

Possible starting points, which we have not verified and which should not limit your search:
- **2a:** Rodinia, Parboil, SHOC, PolyBench/GPU, Lonestar-GPU, Pannotia, and the workload set used by the Accel-Sim authors.
- **2b:** DOE/ECP proxy apps, and molecular dynamics, lattice QCD, CFD, solvers, climate and bioinformatics codes.
- **2c:** published kernel libraries and MLPerf-related kernels.
- **1a and 1b:** papers citing ClusterFusion, CREDIT or FlashFuser, and CUDA projects that use `__cluster_dims__`,
  `map_shared_rank`, `shared::cluster` or `cp.async.bulk`.

## 5. Hard requirements

Reject a candidate that fails any of these, and list it in the rejected table:

1. **Artifact available.** Public code that we can obtain. **Open the repository and confirm the kernels are actually there**
   and are not an empty placeholder or "code coming soon".
2. **Published**, ideally in a top-tier venue of its own field (for example ISCA, MICRO, HPCA, ASPLOS, SC, PPoPP, MLSys,
   NeurIPS, ICML, or the leading journals in physics, chemistry or biology). **Exception:** for class 2a, a workload need not
   have its own paper if it comes from a well-accepted benchmark suite. Preprints do not meet this bar. Report a very strong
   one only if you label it clearly as a preprint.
3. **Runs on a single H100.** No multi-GPU or multi-node requirement, fits in 94 GB, and targets `sm_90` or earlier. Code
   that exists only for Blackwell is out. If it supports both single and multiple GPUs, report it and state the single-GPU
   configuration.
4. **Input data is obtainable**, or the code can generate it.
5. **For classes 1a and 1b, it really uses clusters and DSMEM.** Cite the file in the repository that shows it. A paper
   claiming DSMEM use is not enough.

## 6. Record these, but do not filter on them

These fields decide what can move to Vortex later. Do not reject a workload because of them.

- **Language and API:** CUDA, OpenCL, HIP, SYCL or other. **Does an OpenCL or HIP version exist?** Vortex's README lists
  OpenCL 1.2, Vulkan and HIP, not CUDA, so this is important.
- **Hopper-specific code:** inline PTX, tensor cores (`wgmma`, `mma`), TMA (`cp.async`), `mbarrier`, clusters.
- **Host stack:** standalone binary, or dependent on Python, torch, a JIT, or a large framework.
- **Can the problem be scaled?** Which parameter scales it, and what is the smallest meaningful size? We will need to run
  these on a far slower simulator, so a code that only runs at full size is a real cost.
- **Expected kernel time on an H100-class GPU**, if stated anywhere. Say where the number comes from.
- **A repeated region suitable for sampling:** a timestep, an iteration or a repeated kernel.
- **Licence**, **date of the last commit**, and any **evidence that it builds** today (CI, issues, your reading of the build
  files).

Write "unknown" if you cannot find a value. Do not guess.

## 7. Output format

Produce one Markdown document:

1. **Summary:** counts per class, the most promising candidates, and which classes are still thin.
2. **One table per class** with these columns:

   `Workload | Kernels or region | Paper title | Venue and year | Paper URL | Repo URL | Commit checked | Language/API | OpenCL/HIP version? | Hopper-specific code | Host stack | Scaling parameter and smallest size | Expected H100 time | Repeated region | Licence | Last commit | Evidence it builds | Memory behaviour | Confidence`

   For classes 1a and 1b, add `DSMEM use` (which collective or pattern) and `File proving it`. "Memory behaviour" is one
   short line on which levels it should stress, marked as your expectation.
3. **Rejected candidates:** workload, URL, the requirement it failed, and the evidence.
4. **Open questions** for us to decide.

## 8. Rules for honesty

- **Do not invent URLs, titles, venues or numbers.** Every URL must be one you actually opened.
- Mark anything you could not verify as **unverified**.
- Separate what the source states from what you infer.
- Label preprints as preprints.
- Quality comes before quantity: 15 verified workloads are better than 50 unchecked ones.
