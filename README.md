# Awesome_LLM_Accelerate-PaperList
Since the emergence of chatGPT in 2022, the acceleration of Large Language Model has become increasingly important. Here is a list of papers on accelerating LLMs, currently focusing mainly on inference acceleration, and related works will be gradually added in the future. Welcome contributions!

## Survey
1. Full Stack Optimization for Transformer Inference: a Survey[Paper]
2. A survey of techniques for optimizing transformer inference[Paper]
3. A Survey on Model Compression for Large Language Models[Paper]

## Framework
1. DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale[Paper]
2. TurboTransformers: An Efficient GPU serving System For Transformer Models[Paper]
3. (vLLM)Efficient Memory Management for Large Language Model Serving with PagedAttention[Paper]
4. Fast Distributed Inference Serving for Large Language Models[Paper]
5. AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving[Paper]
6. Orca: A Distributed Serving System for Transformer-Based Generative Models[Paper]
7. Tensor RT-LLM[Repo]
8. lightllm[Repo]
9. LMDeploy[Repo]
10. Text-Generation-Inference[Repo]
## Transformer accelerate
1. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness[Paper]
2. FlashAttention2: Faster Attention with Better Parallelism and Work Partitioning[Paper]
3. FlashDecoding++: Faster Large Language Model Inference on GPUs[Paper]
4. FlashFFTConv: Efficient Convolutions for Long Sentences with Tensor Cores[Paper]
5. FLAT: An Optimized Dataflow for Mitigating Attention Bottlenecks[Paper]
6. ByteTransformer: A High-Performance Transformer Boosted for Variable-Length Inputs[Paper]
7. Fast Transformer Decoding: One Write-Head is All You Need[Paper]
8. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints[Paper]
9. FasterTransformer[Repo]
10. Flash-Decoding for Long-Context Inference[Blog]
11. Accelerating Transformer Networks through Recomposing Softmax Layers[Paper]
12. FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU[Paper]
13. LightSeq: A High Performance Inference Library for Transformers[Paper][Repo] NAACL 2021
14. LightSeq2: LightSeq2: Accelerated Training for Transformer-based Models on GPUs[Paper]
## Model Compression
### Quant
1. Atom: Low-bit Quantization for Efficient and Accurate LLM Serving[Paper]
### Punrning/sparisity
### Low rank
### Communication
1. Overlap communication with dependent compuation via Decompostion in Large Deep Learning Models[Paper]
2. Efficiently scaling Transformer inference[Paper]
## Energy
1. Zeus: Understanding and Optimizing GPU energy Consumption of DNN Training[Paper]