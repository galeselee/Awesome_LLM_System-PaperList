---
description: >-
  This space I will share my reading in. If the theme of paper conforms to the
  paperlist I have organized, I will add it into the paperlist.
---

# Daily reading

### 04/18/2024

[MuxServe: Flexible Multiplexing for Efficient Multiple LLM Serving](https://arxiv.org/abs/2404.02015)

Publish: Arxiv

#### Abstract

At present, when serving multiple LLM models, the main methods are parallelization through spatial (aplaserve) or temporal (deploying each LLM independently). However, given the different numbers of requests for different models and the distinct characteristics of the prefill and decode phases, neither spatial nor temporal parallelization alone can effectively utilize current GPUs. This article attempts to use the MPS method to increase hardware utilization during multi-LLMs serving.

#### Shining point

The observation "given the different numbers of requests for different models and the distinct characteristics of the prefill and decode phases, neither spatial nor temporal parallelization alone can effectively utlize GPU" is a very interesting point.
