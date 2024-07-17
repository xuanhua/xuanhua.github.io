---
layout: post
title:  "Install nvidia-docker on ubuntu 20.04"
date:   2024-07-04 10:29:33 +0000
categories: jekyll update
---

You should first check previous blog [**Install Docker on Ubuntu 20.04**]({% post_url 2024-07-04-install-docker-on-ubuntu %}) to install docker on ubuntu 20.04, then you can follow these steps. But before that, you need to figure out several concepts about these nvidia software packages:

* nvidia driver vs cuda toolkit

  nvidia driver is a software stack that allows the computer to use NVIDIA GPUs, while CUDA Toolkit provides a set of development tools for creating applications that take advantage of NVIDIA GPUs. (this sentence is from my copilot's auto-completion :) And it looks fine)

* nvidia cuda toolkit vs nvidia container toolkit

  These two concepts are related to the use of NVIDIA GPUs in Docker containers, and they have different purposes: CUDA Toolkit is for running CUDA applications directly on GPU hardware, while nvidia container toolkit provides a way to run Docker containers with access to NVIDIA GPUs. (also provided by copilot)

* nvidia docker vs nvidia container toolkit

  nvidia docker is a relative old project. Now it is recommended to use nvidia container toolkit instead of nvidia docker officially. There still nvidia docker supported, but it is named as nvidia docker2 project. Anyway, nvidia container toolkit is what you need if you want to use NVIDIA GPUs in Docker containers. (copilot again)

![nvidia container architecture](https://cloud.githubusercontent.com/assets/3028125/12213714/5b208976-b632-11e5-8406-38d379ec46aa.png)

Here is the steps to install and configure nvidia container toolkit:

* Step 1

  ```bash
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  ```

* Step 2

  ```bash
  sudo apt-get update
  ```

* Step 3

  ```bash
  sudo apt-get install -y nvidia-container-toolkit
  ```

* Step 4

  Configure the container runtime by using the `nvidia-ctk` command:

  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker
  ```

  And then, restart the docker daemon:

  ```bash
  sudo systemctl restart docker
  ```

To verify the installation, you can run `nvidia-smi` inside a container like this:

```bash 
docker run --gpus all  nvidia/cuda:12.4.1-devel-ubuntu20.04  nvidia-smi
```

You should see the GPU information printed out if everything is working correctly.
```text
==========
== CUDA ==
==========

CUDA Version 12.4.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

Fri Jul  5 08:33:33 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:18:00.0 Off |                  N/A |
|  0%   27C    P8              6W /  370W |      16MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 3090        Off |   00000000:3B:00.0 Off |                  N/A |
|  0%   26C    P8              8W /  370W |      16MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA GeForce RTX 3090        Off |   00000000:86:00.0 Off |                  N/A |
|  0%   28C    P8              6W /  370W |      16MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA GeForce RTX 3090        Off |   00000000:AF:00.0 Off |                  N/A |
|  0%   28C    P8             14W /  370W |      16MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
```

Now you can run a container with all GPUs like this:

```bash
docker run --gpus all   --runtime=nvidia  -itd f82e92565c4a bash
```

Here are some explanations about these options:

* `--runtime` option is used to let docker container use nvida provied runtime instead of default docker engine.
* `--gpus` option is used to specify the GPUs that docker can use for this container. The value "all" means all available GPUs.
* `-itd` is the conventional options for running a docker container in interactive mode with pseudo terminal and detached mode. So that you can use composed keys: Ctrl+P, Ctrl+Q to detach the container without stopping it. And you can use `docker attach <container_id>` to reattach the container.

And also if you need to connect the container with a proxy server provided by the host machine, you can add an extra option `--add-host=host.docker.internal:host-gateway` to the docker run command. Then you can use `host.docker.internal` as the hostname in your container to access host machine services (instead of localhost or 127.0.0.1).

And for Docker on Linux there is also an alternative option: `--network="host"` for docker run command. After adding this option for docker run, `127.0.0.1` within the container will point to your docker host. (**Note**: after testing `--network="host"` works better in my case)

So the final command would look like this:

```bash
docker run --gpus all   --runtime=nvidia --network="host"  -itd f82e92565c4a bash
```

**Reference:**

* [A Beginner’s Guide to NVIDIA Container Toolkit on Docker](https://medium.com/@u.mele.coding/a-beginners-guide-to-nvidia-container-toolkit-on-docker-92b645f92006)
* [From inside of a Docker container, how do I connect to the localhost of the machine?](https://stackoverflow.com/questions/24319662/from-inside-of-a-docker-container-how-do-i-connect-to-the-localhost-of-the-mach)