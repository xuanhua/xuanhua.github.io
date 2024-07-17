---
layout: post
title:  "Paper reading notes"
date:   2024-07-05 10:29:33 +0000
categories: jekyll update
---


# Papers

## [FLAME : Factuality-Aware Alignment for Large Language Models](https://arxiv.org/pdf/2405.01525)

**Authors**
Sheng-Chieh Lin1∗, Luyu Gao2, Barlas Oguz3,
Wenhan Xiong3, Jimmy Lin1, Wen-tau Yih3, and Xilun Chen3†
University of Waterloo1, Carnegie Mellon University2, **Meta AI3**

**Abstract**
better quality data
(in terms of factuality) for SFT and DPO does not
necessarily yield models with better factual align-
ment. This is likely because the supervision from
RAG contains information unknown to the LLM;
thus, fine-tuning on RAG generated responses may
inadvertently encourage the LLM to output unfa-
miliar information. To avoid unknown knowledge
from being presented to the LLM, a viable strategy
is to create SFT and DPO training data using the
generated responses from the LLM itself.

**Dataset & Evaluation**

* Alpaca Eval, it comes from paper: "Alpaca-farm: A simulation framework for methods that learn from human feedback"

* `alpaca_eval_gpt4_turbo_fn`

## [OpenELM: An Efficient Language Model Family with Open Training andInference Framework](https://arxiv.org/pdf/2404.14619)

**Authors**
Sachin Mehta Mohammad Hossein Sekhavat Qingqing Cao Maxwell Horton
Yanzi Jin Chenfan Sun Iman Mirzadeh Mahyar Najibi Dmitry Belenko
Peter Zatloukal Mohammad Rastegari **Apple**

**Abstract**
OpenELM is also an fully open-sourced LLM besides OLMo. Also a 3B level model. But it use 2x fewer pre-training tokens and exhibits a 2.36% improvement in an accuracy compared to OLMo. It is trained by an framework called [corenet](https://github.com/apple/corenet), which was built on pytorch.

**Dataset & Evaluation**

* LM Evaluation Harness
  - eo Gao, Jonathan Tow, Stella Biderman, Sid Black, An-thony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, Sept. 2021

* Standard zero-shot tasks
  - [ARC easy and challenge](https://arxiv.org/abs/1803.05457)
  - [BoolQ](https://arxiv.org/abs/1905.10044)
  - [HellaSwag](https://arxiv.org/abs/905.07830)
  - [PIQA]
  - [SciQ]
  - [WinoGrande]

* OpenLLM leaderboard tasks
* LLM360 leaderboard tasks
