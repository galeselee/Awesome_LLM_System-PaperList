---
description: >-
  Since the emergence of chatGPT in 2022, the acceleration of Large Language
  Model has become increasingly important. Here is a list of papers on LLMs
  inference and serving.
---

# Awesome\_LLM\_System-PaperList

### Survey

|                                                                 Paper                                                                |                                   Keywords                                  |   Institute (first)   |           Publication           |                        Others                       |
| :----------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------: | :-------------------: | :-----------------------------: | :-------------------------------------------------: |
|                  [Full Stack Optimization for Transformer Inference: a Survey](https://arxiv.org/pdf/2302.14017.pdf)                 |                       Hardware and software co-design                       |          UCB          |              Arxiv              |                                                     |
|  [A survey of techniques for optimizing transformer inference](https://www.sciencedirect.com/science/article/pii/S1383762123001698)  |                           Transformer optimization                          | Iowa State Univeristy | Journal of Systems Architecture |                                                     |
|                    [A Survey on Model Compression for Large Language Models](https://arxiv.org/pdf/2308.07633.pdf)                   |                              Model Compression                              |          UCSD         |              Arxiv              |                                                     |
| [Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/abs/2312.15234v1) | Optimization technique: quant, pruning, continuous batching, virtual memory |          CMU          |              Arxiv              |                                                     |
|                    [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/abs/2402.16363)                    |                             Performance analysis                            |     Infinigence-AI    |              Arxiv              | [LLMViewer](https://github.com/hahnyuan/LLM-Viewer) |
| [LLM Inference Serving: Survey of Recent Advances and Opportunities](https://arxiv.org/abs/2407.12391v1) | | Northeastern University | Arxiv | |
| [Efficient Large Language Models: A Survey](https://arxiv.org/pdf/2312.03863) | | The Ohio State University | Transactions on Machine Learning Research | |

### Framework

|                                                           Paper/OpenSource Project                                                           |         Keywords         |              Institute (first)             | Publication |                         Others                        |
| :------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------: | :----------------------------------------: | :---------: | :---------------------------------------------------: |
| [DeepSpeed Infernce: Enabling Efficient Inference of Transformer Models at Unprecedented Scale](https://export.arxiv.org/pdf/2207.00032.pdf) | Deepspeed; Kerenl Fusion |                  MicroSoft                 |   SC 2022   | [Github repo](https://github.com/microsoft/DeepSpeed) |
|      [DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference](https://arxiv.org/pdf/2401.08671.pdf)     |   Deepspeed; Split fuse  |                  MicroSoft                 |    Arxiv    | [Github repo](https://github.com/microsoft/DeepSpeed) |
|           [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180.pdf)           |   vLLM; pagedAttention   |                     UCB                    |  SOSP 2023  |  [Github repo](https://github.com/vllm-project/vllm)  |
|                                    [TensorRT-LLM/FastTransformer](https://github.com/NVIDIA/TensorRT-LLM)                                    |                          |                   NVIDIA                   |             |                                                       |
|                                                [lightLLM](https://github.com/ModelTC/lightllm)                                               |                          | Shanghai Artifcial Intelligence Laboratory |             |                                                       |
|                                                 [MLC LLM](https://github.com/mlc-ai/mlc-llm)                                                 |   TVM; Multi-platforms   |                  MLC-Team                  |             |                                                       |
|                          [Text-Generation-Inference(TGI)](https://github.com/huggingface/text-generation-inference)                          |                          |                 Huggingface                |             |                                                       |

### Serving

|                            Paper                             |                           Keywords                           |       Institute (first)       | Publication  |                            Others                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :---------------------------: | :----------: | :----------------------------------------------------------: |
| [Fast Distributed Inference Serving for Large Language Models](https://arxiv.org/pdf/2305.05920.pdf) |                Distributed inference serving                 |              PKU              |    Arxiv     |                                                              |
| [AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving](https://arxiv.org/abs/2302.11665/pdf) |               Pipeline Parallel; Auto parallel               |              UCB              |  OSDI 2023   |     [Github repo](https://github.com/alpa-projects/mms)      |
| [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf) |                     Continuous batching                      |   Seoul National University   |   OSDI2022   |                                                              |
| [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) |                   Multiple Decoding Heads                    |     Princeton University      |    Arxiv     |   [Github repo](https://github.com/FasterDecoding/Medusa)    |
| [PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU](https://arxiv.org/abs/2312.12456) |                      Consumer-grade GPU                      |             SJTU              |    Arxiv     |   [Github repo](https://github.com/SJTU-IPADS/PowerInfer)    |
| [LLM in a flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/abs/2312.11514) |                        flash; Pruning                        |             Apple             |    Arxiv     |                                                              |
| [Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline](https://arxiv.org/abs/2305.13144) |                      Length Perception                       |              NUS              | NeurIPS 2023 | [Github repo](https://github.com/zhengzangw/Sequence-Scheduling) |
| [S3: Increasing GPU Utilization during Generative Inference for Higher Throughput](https://arxiv.org/abs/2306.06000) |                                                              |      Harvard University       |    Arxiv     |                                                              |
| [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670) |                           Decouple                           |              PKU              |  OSDI 2024   |                                                              |
| [Splitwise: Efficient generative LLM inference using phase splitting](https://arxiv.org/abs/2311.18677) |                           Decouple                           |              UW               |  ISCA 2024   | [Track issue](https://github.com/vllm-project/vllm/issues/2472) |
| [Efficiently Programming Large Language Models using SGLang](https://arxiv.org/abs/2312.07104) |                        Agent Language                        |              UCB              |    Arxiv     |     [Github repo](https://github.com/sgl-project/sglang)     |
| [FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](https://arxiv.org/abs/2303.06865) |                          Single GPU                          |      Stanford University      |    Arxiv     |    [Github repo](https://github.com/FMInference/FlexGen)     |
| [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) |                           Decouple                           |            GaTech             |  OSDI 2024   |                                                              |
| [SpotServe: Serving Generative Large Language Models on Preemptible Instances](https://arxiv.org/abs/2311.15566) |                       Preemptible GPU                        |              CMU              | ASPLOS 2024  |   [Empty Github repo](https://github.com/hsword/spotserve)   |
| [SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification](https://arxiv.org/abs/2305.09781) |                    Tree-based Speculative                    |              CMU              | ASPLOS 2024  |                                                              |
| [AttentionStore: Cost-effective Attention Reuse across Multi-turn Conversations in Large Language Model Serving](https://arxiv.org/abs/2403.19708) |  Cache the multi-turn prefill KV-cache in host-DRAM and SSD  |              NUS              |    ATC 2024     |                                                              |
| [MuxServe: Flexible Multiplexing for Efficient Multiple LLM Serving](https://arxiv.org/pdf/2404.02015.pdf) | Use spatial-temporal multiplexing method to serve multi-LLMs |             MMLab             |    Arxiv     |                                                              |
| [PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inference](https://arxiv.org/abs/2405.12532) |                     KV Cache Compression                     | Shanghai Jiao Tong University |    Arxiv     |                                                              |
| [You Only Cache Once: Decoder-Decoder Architectures for Language Models](https://arxiv.org/abs/2405.05254) |                           KV Cache                           |      Microsoft Research       |    Arxiv     |                                                              |
| [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) |                    Multi-token Prediction                    |             Meta              |    Arxiv     |                                                              |
| [ExeGPT: Constraint-Aware Resource Scheduling for LLM Inference](https://arxiv.org/abs/2404.07947) |      Decouple         |   Hanyang University     |    ASPLOS 2024     |                   |
| [Parrot: Efficient Serving of LLM-based Applications with Semantic Variable](https://arxiv.org/abs/2405.19888) |    LLM Applications        |   SJTU    |    OSDI 2024     |                   |
| [Fairness in Serving Large Language Models](https://arxiv.org/abs/2401.00588) | Fairness; LLM Serving | UC Berkeley,Stanford University | OSDI 2024 |  |
| [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://github.com/kvcache-ai/Mooncake/blob/main/Mooncake-v1.pdf) |                           KV Cache                           |          Moonshot AI          |    GitHub    |                                                              |
| [MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention](https://arxiv.org/pdf/2407.02490) | Pre-fillingfor Long-Context<br />Dynamic Sparse Attention | Microsoft | Arxiv | [Github repo](https://github.com/microsoft/MInference?tab=readme-ov-file) |
| [MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool](https://arxiv.org/abs/2406.17565) | Memory Pool | Huawei | Arxiv | |
| [InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management](https://www.usenix.org/conference/osdi24/presentation/lee) | sparisity | Seoul National University | OSDI 2024 | |
| [Llumnix: Dynamic Scheduling for Large Language Model Serving](https://www.usenix.org/conference/osdi24/presentation/sun-biao) | Preemptible GPU | Alibaba Group | OSDI 2024 | |
| [PUZZLE: Efficiently Aligning Large Language Models through Light-Weight Context Switch](https://www.usenix.org/conference/atc24/presentation/lei) | Multi-Agent | Tsinghua University ｜ ATC 2024 | ||
| [SampleAttention: Near-Lossless Acceleration of Long Context LLM Inference with Adaptive Structured Sparse Attention](https://arxiv.org/abs/2406.15486) | Sparsity; Long context | PKU | Arxiv |  |
| [Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference](https://arxiv.org/abs/2406.10774) | Sparsity; Related token | MIT | ICML 2024 | |
| [Accelerating Production LLMs with Combined Token/Embedding Speculators](https://arxiv.org/abs/2404.19124) | Speculative decoding | IBM Research | Arxiv | [Github repo](https://github.com/foundation-model-stack/fms-fsdp) |
| [LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference](https://arxiv.org/abs/2407.14057) | KV Cache | Apple | Arxiv |  |


### Operating System

|                                 Paper                                |    Keywords   |  Institute(first)  | Publication | Others |
| :------------------------------------------------------------------: | :-----------: | :----------------: | :---------: | :----: |
| [AIOS: LLM Agent Operating System](https://arxiv.org/abs/2403.16971) | OS; LLM Agent | Rutgers University |    Arxiv    |        |

### Transformer accelerate

|                            Paper                             |              Keywords              |        Institute (first)        | Publication  |                            Others                            |
| :----------------------------------------------------------: | :--------------------------------: | :-----------------------------: | :----------: | :----------------------------------------------------------: |
| [TurboTransformers: An Efficient GPU serving System For Transformer Models](https://arxiv.org/pdf/2010.05680.pdf) |                                    |             Tencent             |  PPoPP 2021  | [Github repo](https://github.com/Tencent/TurboTransformers)  |
| [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) |   FlashAttention; Online Softmax   |       Stanford University       | NeurIPS 2023 | [Github repo](https://github.com/Dao-AILab/flash-attention)  |
| [FlashAttention2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) |                                    |       Stanford University       |    Arxiv     | [Github repo](https://github.com/Dao-AILab/flash-attention)  |
| [FlashDecoding++: Faster Large Language Model Inference on GPUs](https://arxiv.org/abs/2311.01282) | Softmax with Unified Maximum Value |       Tsinghua University       |  Mlsys 2024  |                                                              |
| [FlashFFTConv: Efficient Convolutions for Long Sentences with Tensor Cores](https://arxiv.org/abs/2311.05908) |  FFT; TensorCore; Long Sentences   |       Stanford University       |    Arxiv     | [Github repo](https://github.com/HazyResearch/flash-fft-conv) |
| [FLAT: An Optimized Dataflow for Mitigating Attention Bottlenecks](https://arxiv.org/abs/2107.06419) |                                    | Georgia Institute of Technology | ASPLOS 2023  |                                                              |
| [ByteTransformer: A High-Performance Transformer Boosted for Variable-Length Inputs](https://arxiv.org/pdf/2210.03052.pdf) |       Variable-Length Inputs       |               UCR               |  PPoPP 2022  | [Github repo](https://github.com/bytedance/ByteTransformer)  |
| [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) |                MQA                 |             Google              |    Arxiv     |                                                              |
| [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245.pdf) |                GQA                 |         Google Research         |   ACL 2023   |                                                              |
| [LightSeq: A High Performance Inference Library for Transformers](http://arxiv.org/abs/2010.13887) |                                    |            ByteDance            |  NAACL 2021  |     [Github repo](https://github.com/bytedance/lightseq)     |
| [LightSeq2: LightSeq2: Accelerated Training for Transformer-based Models on GPUs](https://arxiv.org/pdf/2110.05722.pdf) |                                    |            ByteDance            |   SC 2022    |                                                              |
| [Blockwise Parallel Transformer for Large Context Models](https://arxiv.org/pdf/2305.19370.pdf) |       Blockwise transformer        |               UCB               | NeurIPS 2023 | [Github repo](https://github.com/kyegomez/Blockwise-Parallel-Transformer) |
| [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](https://arxiv.org/abs/2405.04437) |     Dynamic Memory Management      |    Microsoft Research India     |    Arxiv     |                                                              |

### Model Compression

#### Quant

|                                                    Paper                                                    |       Keywords      | Institute (first) | Publication |                     Others                     |
| :---------------------------------------------------------------------------------------------------------: | :-----------------: | :---------------: | :---------: | :--------------------------------------------: |
|     [Atom: Low-bit Quantization for Efficient and Accurate LLM Serving](http://arxiv.org/abs/2310.19102)    |                     |        SJTU       |    mlsys 2024    | [Github repo](https://github.com/efeslab/Atom) |
| [Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference](https://arxiv.org/abs/2403.09636) | Dynamic Compression |       NVIDIA      |    Arxiv    |                                                |
| [Quant-LLM: Accelerating the Serving of Large Language Models via FP6-Centric Algorithm-System Co-Design on Modern GPUs](https://www.usenix.org/conference/atc24/presentation/xia) | FP6 | USYD | ATC 2024 | |
| [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) | AWQ | MIT | mlsys 2024 bp| |

#### Punrning/sparisity

|                            Paper                             |              Keywords               |       Institute (first)       | Publication |                            Others                            |
| :----------------------------------------------------------: | :---------------------------------: | :---------------------------: | :---------: | :----------------------------------------------------------: |
| [Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity](https://arxiv.org/abs/2309.10285) |                                     |     Univeristy of Sydney      |  VLDB 2024  | [Github repo](https://github.com/AlibabaResearch/flash-llm)  |
| [CLLMs: Consistency Large Language Models](https://arxiv.org/abs/2403.00835) |             Consistency             | Shanghai Jiao Tong University |    Arxiv    | [Github repo](https://github.com/hao-ai-lab/Consistency_LLM) |

### Communication

|                                                                         Paper                                                                        |         Keywords        | Institute (first) | Publication | Others |
| :--------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------: | :---------------: | :---------: | :----: |
| [Overlap communication with dependent compuation via Decompostion in Large Deep Learning Models](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959) |         Overlap         |       Google      | ASPLOS 2023 |        |
|                                     [Efficiently scaling Transformer inference](https://arxiv.org/abs/2211.05102)                                    |         Scaling         |       Google      |  Mlsys 2023 |        |
| [Centauri: Enabling efficient scheduling for communication-computation overlap in large model training via communication](https://dl.acm.org/doi/10.1145/3620666.3651379) | communication partition |        PKU        | ASPLOS 2024 |        |

### Energy

|                                                              Paper                                                              | Keywords | Institute (first) | Publication |                      Others                      |
| :-----------------------------------------------------------------------------------------------------------------------------: | :------: | :---------------: | :---------: | :----------------------------------------------: |
| [Zeus: Understanding and Optimizing GPU energy Consumption of DNN Training](https://www.usenix.org/system/files/nsdi23-you.pdf) |          |  Yale University  |  NSDI 2023  | [Github repo](https://github.com/ml-energy/zeus) 
| [Power-aware Deep Learning Model Serving with μ-Serve](https://www.usenix.org/system/files/atc24-qiu.pdf) | | UIUC | ATC 2024 | |

### Decentralized

|                                                             Paper                                                            |      Keywords      | Institute (first) | Publication | Others |
| :--------------------------------------------------------------------------------------------------------------------------: | :----------------: | :---------------: | :---------: | :----: |
| [FusionAI: Decentralized Training and Deploying LLMs with Massive Consumer-Level GPUs](https://arxiv.org/pdf/2309.01172.pdf) | Consumer-grade GPU |        HKBU       |    Arxiv    |        |
|              [Petals: Collaborative Inference and Fine-tuning of Large Models](https://arxiv.org/abs/2209.01188)             |                    |       Yandex      |    Arxiv    |        |

### Serveless

|                                                             Paper                                                            |      Keywords      | Institute (first) | Publication | Others |
| :--------------------------------------------------------------------------------------------------------------------------: | :----------------: | :---------------: | :---------: | :----: |
| [ServerlessLLM: Locality-Enhanced Serverless Inference for Large Language Models](https://arxiv.org/abs/2401.14351) | cold boot |        The University of Edinburgh       |    OSDI 2024    | [Empty Github](https://github.com/ServerlessLLM/ServerlessLLM)  |
| [StreamBox: A Lightweight GPU SandBox for Serverless Inference Workflow](https://www.usenix.org/conference/atc24/presentation/wu-hao) | | HUST | ATC 2024 | [Github](https://github.com/CGCL-codes/streambox) |


### Trace
|                                                             Paper                                                            |      Keywords      | Institute (first) | Publication | Others |
| :--------------------------------------------------------------------------------------------------------------------------: | :----------------: | :---------------: | :---------: | :----: |
| [Characterization of Large Language Model Development in the Datacenter](https://arxiv.org/abs/2403.07648) | Cluster trace(for LLM) | ShangHai AI Lab |    NSDI 2024    | [Github](https://github.com/InternLM/AcmeTrace)  |
| [BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems](https://arxiv.org/abs/2401.17644) | GPT users trace | HKUSTGZ | Arxiv 2024 | [Github](https://github.com/HPMLL/BurstGPT/)|

