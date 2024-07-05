---
layout: post
title:  "Install nvidia-docker on ubuntu 20.04"
date:   2024-07-04 10:29:33 +0000
categories: jekyll update
---

you should first check previous blog [Install Docker on Ubuntu 20.04] to install docker on ubuntu 20.04, then you can follow these steps:

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
