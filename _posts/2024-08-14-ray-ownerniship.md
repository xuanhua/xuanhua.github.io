---
layout: post
title:  "Ownership: A Distributed Futures System for Fine-Grained Tasks"
date:   2024-08-14 10:29:33 +0000
categories: jekyll update
---

This paper give a design of distributed futures system. But for users, we just need to understand the most significant concepts.

**What is a `Future` in python language?**

```text
A Future represents an eventual result of an asynchronous operation. Not thread-safe.
```
[Click for more details ](https://docs.python.org/3/library/asyncio-future.html)

But what is a distributed future ? We do not provide the literal answer now.

Let's start from an examle: a distributed system that compute $c = Task_a() + Task_b()$. This is a simplification of many real world tasks. From the distributed perspective, it could be done for various ways as shown in below picture.

```python
a_future = compute()
b_future = compute()
c_future = add(a_future, b_future)
c = system.get(c_future)
```

![fig 2](/assets/img/2024-08-15-ray-ownership-fig2.png)



Suppose we have three processes: `Driver`, `Worker 1` and `Worker 2`. The Driver is the client that sends tasks to Workers. Each task has a unique ID (TaskID). When a worker completes its computation for a given TaskID, it notifies the driver by sending back the result of the computation along with the TaskID. There are 4 approaches as shown in above figure. ($T_i$ means task $i$).

- a) Without any parallel, all tasks are done sequentially.
- b) Worker 1 handle $T_a$ and worker 2 handle $T_b$, the only difference from last step is result $a$ is sent to worker 2. Thus transfering result $b$ to driver are eliminated.
- c）$T_a$ and $T_b$ are computed in parallel by worker 1 and worker 2 respectively. The results of both computations are sent back to driver.
- d) Almost the same with last step, but result $a$ is not sent back to driver but to worker 2. Thus this is the most efficient way.

A fine-grained distributed computing system become more and more important nowdays. At least better than previous hashing based approaches.

It is practical to execute millions of fine-grained functions for below cases:
* Reinforcement learning
* Video processing
* Model serving

**What is a distributed futures system?**

Still let's see an example. In a typical image classification scenario. An image in a request comes through various precprocessing and queued together to be processed at a batch. The batch is then sent off for inference to a GPU-based model.

In Ray,  there are *Futures* and there are also *Actors*.

Actor is a stateful object that can send messages to other actors and maintain its state over time. But how does it work? Let' continue.

In the case of model serving, the goal is to reduce request latency while maximizing throughput often by using model replicas. Let's check how *Actor* work in this case.

![ray model serving](/assets/img/2024-08-15-ray-model-serving.png)

In this example
* A lot of `Preprocess` could be handled in parallel.
* Single `Request` are queued up in `Router` to become a batch to be handled all together. 
* `Model` is usually a GPU-based computing task.
* In this case, `Preprocess` are all represented as `Tasks` that is without state; `Router` and `Model` are actually `Actors`, that could keep model weights and queued requests;
* Distributed futures as a reference of preprocessed images could be passed to Model actors, instread of copying actually image data.

**References**

* [Ownership: A Distributed Futures System
for Fine-Grained Tasks]()
* [Ray 1.x Architecture](https://docs.google.com/document/d/1lAy0Owi-vPz2jEqBSaHNQcy2IBSDEHyXNOQZlGuj93c/preview)



