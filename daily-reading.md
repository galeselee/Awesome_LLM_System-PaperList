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



### 04/28/2024

[Fairness in Serving Large Language Models](https://arxiv.org/abs/2401.00588)

Publish: OSDI 2024

#### Abstract

Regarding the current server's selection of the next request to execute, the typical strategies used are First-Come, First-Served (FCFS) or RPM (request per minute). However, these strategies can lead to various issues. For instance, FCFS can result in clients that send requests more frequently being prioritized over clients that send requests less frequently, while RR can lead to low machine utilization rates (Maybeit might a decent business strategy?). The article give a fairness definition in LLM serving senario and provide a scheduling strategy based on Virtual Token Counter, which aims to make the server's processing of different types of clients' requests more equitable.

\


