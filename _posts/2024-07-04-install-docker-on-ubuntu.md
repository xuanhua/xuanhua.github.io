---
layout: post
title:  "Install docker-ce on ubuntu 20.04"
date:   2024-07-04 10:29:33 +0000
categories: jekyll update
---


This blog records steps of installing docker-ce on ubuntu 20.04. For people with no patience, I provide the quick solution/steps. (note that: I'm using a copilot during writing this post)

* Install the prerequisites

  ```bash
  sudo apt-get update
  sudo apt install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
  ```

* Install GPG key

  ```bash
  sudo curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
  ```
  Note that I choose aliyun mirror instead of docker official, you can choose other mirrors as well.

* Create source list file for docker-ce

  ```bash
  sudo echo "deb [arch=amd64 signed-by=/etc/apt/trusted.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu focal stable" > /etc/apt/sources.list.d/docker-ce.list
  ```

* Install docker-ce

  ```bash
  sudo apt update
  sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  ```

* Add user to docker group

  ```bash
  # suppose that <username> is the username of your current user
  sudo usermod -aG docker <username>
  newgrp  docker
  ```

  You should login the machine from a new terminal otherwise you might get permission deny for running docker commands as user `<username>`


**Reference:**
* [ubuntu 20.04通过国内源安装docker](https://www.cnblogs.com/amsilence/p/16404609.html)


For other issues I met, I just write down here as FAQ and just FYI:

Q1: Why I always got error "Cannot initiate the connection to download.docker.com:443" for command: `sudo apt update` ?

A: The answer is not sure for me. But some way you might try is to use a proxy server, or you should use another source like aliyun mirror.

Q2: How to make `sudo apt update` use network proxy?

A: Here are two things you can try:

  1) Set environment variables HTTP_PROXY and HTTPS_PROXY in a new file like `my_proxy.conf` and put it under `/etc/apt/apt.conf.d/`, in `my_proxy.conf` with following content:

  ```bash
  Acquire::http::Proxy "http://<proxy_host>:<proxy_port>/";
  Acquire::https::Proxy "https://<proxy_host>:<proxy_port>/";
  ```

  2) Since you are using `sudo`, you should make sure the environment variables `HTTP_PROXY` and `HTTPS_PROXY` are passed in. To make that happen, you need change `/etc/sudoers` file by using `vim /etc/sudoers` and add following changes:

  ```bash
  #Defaults        env_reset
  Defaults       env_keep="http_proxy https_proxy"
  ```

Q3: No permission for command: `sudo echo "deb [arch=amd64 signed-by=/etc/apt/trusted.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu focal stable" > /etc/apt/sources.list.d/docker-ce.list` 

A: As a root user, you could manually create file: `/etc/apt/sources.list.d/docker-ce.list` and add the content to it. You might need sudo permission for that operation. 

**Final words**

Preparing working environment for most of the time is frustrating, especially when you are resisting to understand what's going on behind the scenes. But once you get a hang of it, things become much easier and more manageable. Knowing what root user can do could help you bypass some issues or create your own alternative solutions. For this kind of tasks, there is no skills but only understanding things beneath the surface helps.