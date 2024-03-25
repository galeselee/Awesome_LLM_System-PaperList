# Awesome_LLM_Accelerate-PaperList
Since the emergence of chatGPT in 2022, the acceleration of Large Language Model has become increasingly important. Here is a list of papers on accelerating LLMs, currently focusing mainly on inference acceleration, and related works will be gradually added in the future. Welcome contributions!

## Survey

|  Paper | Keywords|Institute (first)|Publication | Others 
| :------------------: | :--------------: | :-----------: | :---------: | :---------:|
[Full Stack Optimization for Transformer Inference: a Survey](https://arxiv.org/pdf/2302.14017.pdf)|Hardware and software co-design | UCB | Arxiv | |
[A survey of techniques for optimizing transformer inference](https://www.sciencedirect.com/science/article/pii/S1383762123001698) | Transformer optimization |Iowa State Univeristy | Journal of Systems Architecture||
[A Survey on Model Compression for Large Language Models](https://arxiv.org/pdf/2308.07633.pdf)|Model Compression | UCSD | Arxiv
|[Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/abs/2312.15234v1)|Optimization technique: quant, pruning, continuous batching, virtual memory| CMU| Arxiv
|[LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/abs/2402.16363)|Performance analysis| Infinigence-AI | Arxiv | [LLMViewer](https://github.com/hahnyuan/LLM-Viewer)

## Framework 
|  Paper/OpenSource Project | Keywords|Institute (first)|Publication | Others 
| :------------------: | :--------------: | :-----------: | :---------: | :---------:|
[DeepSpeed Infernce: Enabling Efficient Inference of Transformer Models at Unprecedented Scale](https://export.arxiv.org/pdf/2207.00032.pdf)| Deepspeed; Kerenl Fusion | MicroSoft | SC 2022 | [Github repo](https://github.com/microsoft/DeepSpeed)|
[DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference](https://arxiv.org/pdf/2401.08671.pdf)|Deepspeed; Split fuse| MicroSoft | Arxiv |[Github repo](https://github.com/microsoft/DeepSpeed) |
[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180.pdf)| vLLM; pagedAttention | UCB | SOSP 2023|[Github repo](https://github.com/vllm-project/vllm)
[TensorRT-LLM/FastTransformer](https://github.com/NVIDIA/TensorRT-LLM) |  | NVIDIA
[lightLLM](https://github.com/ModelTC/lightllm) | | Shanghai Artifcial Intelligence Laboratory
[MLC LLM](https://github.com/mlc-ai/mlc-llm) | TVM; Multi-platforms | MLC-Team
[Text-Generation-Inference(TGI)](https://github.com/huggingface/text-generation-inference) |  | Huggingface

## Server
|  Paper | Keywords|Institute (first)|Publication | Others 
| :------------------: | :--------------: | :-----------: | :---------: | :---------:|
[Fast Distributed Inference Serving for Large Language Models](https://arxiv.org/pdf/2305.05920.pdf)|Distributed inference serving | PKU | Arxiv |  
[AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving](https://arxiv.org/abs/2302.11665/pdf) | Pipeline Parallel; Auto parallel | UCB | OSDI 2023| [Github repo](https://github.com/alpa-projects/mms)
[Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf) | Continuous batching | Seoul National University | OSDI2022|
[Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) |Multiple Decoding Heads |Princeton University | Arxiv| [Github repo](https://github.com/FasterDecoding/Medusa)
[PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU](https://arxiv.org/abs/2312.12456) | Consumer-grade GPU | SJTU | Arxiv | [Github repo](https://github.com/SJTU-IPADS/PowerInfer)
[LLM in a flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/abs/2312.11514) | flash; Pruning | Apple | Arxiv  | 
[Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline](https://arxiv.org/abs/2305.13144)|Length Perception | NUS | NeurIPS 2023 | [Github repo](https://github.com/zhengzangw/Sequence-Scheduling)
[S3: Increasing GPU Utilization during Generative Inference for Higher Throughput](https://arxiv.org/abs/2306.06000)| | Harvard University | Arxiv | 
[DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670) | Decouple | PKU | OSDI 2024 
[Splitwise: Efficient generative LLM inference using phase splitting](https://arxiv.org/abs/2311.18677) |  Decouple | UW | Arxiv | [Track issue](https://github.com/vllm-project/vllm/issues/2472)
[Efficiently Programming Large Language Models using SGLang](https://arxiv.org/abs/2312.07104) | Agent Language | UCB | Arxiv | [Github repo](https://github.com/sgl-project/sglang)
[FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](https://arxiv.org/abs/2303.06865) | Single GPU | Stanford University | Arxiv | [Github repo](https://github.com/FMInference/FlexGen)
[Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) | Decouple | GaTech | OSDI 2024 | 
[SpotServe: Serving Generative Large Language Models on Preemptible Instances] | Preemptible GPU | CMU | ASPLOS 2024 | [Empty Github repo](https://github.com/hsword/spotserve)
[SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification] | Tree-based Speculative | CMU | ASPLOS 2024 | 


## Transformer accelerate
|  Paper | Keywords|Institute (first)|Publication | Others 
| :------------------: | :--------------: | :-----------: | :---------: | :---------:|
[TurboTransformers: An Efficient GPU serving System For Transformer Models](https://arxiv.org/pdf/2010.05680.pdf)| | Tencent | PPoPP 2021|[Github repo](https://github.com/Tencent/TurboTransformers)
[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)|FlashAttention; Online Softmax | Stanford University| NeurIPS 2023 | [Github repo](https://github.com/Dao-AILab/flash-attention)
[FlashAttention2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) | | Stanford University | Arxiv | [Github repo](https://github.com/Dao-AILab/flash-attention)
[FlashDecoding++: Faster Large Language Model Inference on GPUs](https://arxiv.org/abs/2311.01282) | Softmax with Unified Maximum Value | Tsinghua University | Mlsys 2024 | 
[FlashFFTConv: Efficient Convolutions for Long Sentences with Tensor Cores](https://arxiv.org/abs/2311.05908) | FFT; TensorCore; Long Sentences | Stanford University | Arxiv | [Github repo](https://github.com/HazyResearch/flash-fft-conv)
[FLAT: An Optimized Dataflow for Mitigating Attention Bottlenecks](https://arxiv.org/abs/2107.06419) | |Georgia Institute of Technology |ASPLOS 2023 | 
[ByteTransformer: A High-Performance Transformer Boosted for Variable-Length Inputs](https://arxiv.org/pdf/2210.03052.pdf) | Variable-Length Inputs | UCR | PPoPP 2022 | [Github repo](https://github.com/bytedance/ByteTransformer)
[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) | MQA | Google | Arxiv|
[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245.pdf) | GQA | Google Research | ACL 2023 
[LightSeq: A High Performance Inference Library for Transformers](http://arxiv.org/abs/2010.13887)| |ByteDance | NAACL 2021 | [Github repo](https://github.com/bytedance/lightseq)
[LightSeq2: LightSeq2: Accelerated Training for Transformer-based Models on GPUs](https://arxiv.org/pdf/2110.05722.pdf) |  |ByteDance  | SC 2022 
## Model Compression
### Quant
|  Paper | Keywords|Institute (first)|Publication | Others 
| :------------------: | :--------------: | :-----------: | :---------: | :---------:|
[Atom: Low-bit Quantization for Efficient and Accurate LLM Serving](http://arxiv.org/abs/2310.19102) | |SJTU |Arxiv| [Github repo](https://github.com/efeslab/Atom)
[Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference](https://arxiv.org/abs/2403.09636) | Dynamic Compression | NVIDIA | Arxiv | 


### Punrning/sparisity
|  Paper | Keywords|Institute (first)|Publication | Others 
| :------------------: | :--------------: | :-----------: | :---------: | :---------:|
[Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity](https://arxiv.org/abs/2309.10285) | | Univeristy of Sydney | VLDB 2024 | [Github repo](https://github.com/AlibabaResearch/flash-llm)
## Communication
|  Paper | Keywords|Institute (first)|Publication | Others 
| :------------------: | :--------------: | :-----------: | :---------: | :---------:|
[Overlap communication with dependent compuation via Decompostion in Large Deep Learning Models](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959) | Overlap | Google | ASPLOS 2023 | 
[Efficiently scaling Transformer inference](https://arxiv.org/abs/2211.05102) | Scaling | Google | Mlsys 2023 
## Energy
|  Paper | Keywords|Institute (first)|Publication | Others 
| :------------------: | :--------------: | :-----------: | :---------: | :---------:|
[Zeus: Understanding and Optimizing GPU energy Consumption of DNN Training](https://www.usenix.org/system/files/nsdi23-you.pdf) | | Yale University | NSDI 2023 | [Github repo](https://github.com/ml-energy/zeus)
## Decentralized
|  Paper | Keywords|Institute (first)|Publication | Others 
| :------------------: | :--------------: | :-----------: | :---------: | :---------:|
[FusionAI: Decentralized Training and Deploying LLMs with Massive Consumer-Level GPUs](https://arxiv.org/pdf/2309.01172.pdf) | Consumer-grade GPU | HKBU | Arxiv | 
[Petals: Collaborative Inference and Fine-tuning of Large Models](https://arxiv.org/abs/2209.01188) | | Yandex | Arxiv

## Serveless
|  Paper | Keywords|Institute (first)|Publication | Others 
| :------------------: | :--------------: | :-----------: | :---------: | :---------:|
[ServerlessLLM: Locality-Enhanced Serverless Inference for Large Language Models](https://arxiv.org/abs/2401.14351) | |The University of Edinburgh | Arxiv | [Empty Github repo](https://github.com/ServerlessLLM/ServerlessLLM)
