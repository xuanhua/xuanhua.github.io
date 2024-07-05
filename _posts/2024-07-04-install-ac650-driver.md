---
layout: post
title:  "Install ac650 driver on ubuntu 20.04"
date:   2024-07-04 10:29:33 +0000
categories: jekyll update
---

Two steps to install ac650 driver on ubuntu 20.04

* Step 1: Download the driver source code from github

```bash
https://github.com/brektrou/rtl8821CU.git
```

* Step 2: Compile and install the driver

Here we will not use `dkms`, otherwise you have to install `dkms` first if it's not installed yet. We just use make command for compiling.

```bash
cd rtl8821CU/
make clean
make
sudo make install
```

* Extra step: If there is no wifi adapter could be seen after installing the driver, you need to reboot your system.


Above are the steps to make your ubuntu 20.04 work with ac650 usb-based wireless network card. Hope it helps. Please let me know if any issues occur.