---
layout: post
title:  "How to use vscode to connect any machine"
date:   2024-09-10 10:29:33 +0000
categories: jekyll update
---

VSCode is a powerful tool that can be used for remote development. You can use your laptop (e.g. your macbook pro) to write and debug code on the remote GPU server by vscode. And you like remote work from home or from starbucks or anywhere you like while you still want to write code on your GPU server just like in your office. It seems possible if you know some networking skills or impossible if you know nothing about it. In both cases, I think my experience may help you on this.

I will state the scenario first then I will give my solutions. 

**Scenario 1** you have a laptop (call it host_A), and host_A could only connect to a jump server (called host_B), the ultimate destination is GPU server (host_C) that your want to access.

**Solution 1 for Scenario 1**

![alt text](/assets/img/2024-09-11-image-1.png)

The first solution is to use 'ssh jump', basically if host_A could access host_B by bash command: `ssh user_on_B@host_B`; And you could also access host_C from host_B by `ssh user_on_C@host_C`. Then you could directly ssh into your GPU server from your laptop using the following 'ssh jump' command:

```bash
# Run this from host_A, make request to host_B first then sent to host_C
ssh -J user_on_B@host_B user_on_C@host_C
```

If this command could work properly. You can add this into your vscode remote connection settings by following steps:

* Open the 'Remote explorer' of vscode, and click on the plus button to create a new SSH target.
![alt text](/assets/img/2024-09-11-image-2.png)

* Enter the previous bash command into the field and press enter 
![alt text](/assets/img/2024-09-11-image-3.png)

* Find this entry in 'Remote explorer', and click 'connect in New Window'.

If everything goes well, you should be able to use vscode connecting to your GPU server directly.

Actually the first two steps did is create an entry in your ssh config file (`~/.ssh/config`):

```bash
# you should replace user_on_*, host_* with your actual username and server IP address.
Host host_C 
  HostName host_C 
  ProxyJump user_on_B@host_B
  User user_on_C
```
You could open `~/.ssh/config` and update the `Host` field to other specific name. 【Note: directly edit this file and add a such entry might not work properly, so I recommend adding such entry from vscode GUI】

**Solution 2 for Scenario 1**

The other solution basically is to use ssh tunnel directly by creating 'Local port forwarding'. Our first solution actually is also kind of ssh tunnels and it provide more intuitive way to operate. But 'Local port forwarding' is also a popular techniques you should know. (Google it if you did not hear much about 'local port forwarding') 

ssh tunnel is a technique that allows your local machine to access various services running on remote server which does not allows connection from outside except ssh. e.g. The remote server provide mongodb service, but did not allow connection other than localhost. You can use ssh tunnel to overcome the restriction and access the database directly from your local machine. 

Let's go back to our solution for scenario 1, the simplest solution is to create a 'local port forwarding' on host_B (say port_F), and then using ssh connection from host_A to host_B with port_F could achieve the accessing directly from host_A to host_C, let's see the solution step by step:

![alt text](/assets/img/2024-09-11-image-4.png)

1) On host_B, create a local port forwarding GPU server(host_C), with following command:
```bash
# -f: Requests ssh to go to background just before command execution. 
# -N: Do not execute a remote command. 
# -C: Specifies that compression will be used when sending commands to the server.
# -L: [bind_address:]port:host:hostport Specifies that the given port on the local (client) host is to be forwarded to the specified host and port on the remote side.
ssh -fCN -L $port_F:$host_C:22 $user_on_C@$host_C
```
And port_F is a arbitrary unused port on host_B (and `LPF_on_B` means 'local port forwarding on host_B')

2) So on host_A, you can use following command to create ssh connection to host_C:
```bash
ssh -p $port_F $user_on_C@$host_B
```
What happens when `ssh -p $port_F $user_on_C@$host_B` includes:

a) On host_A, there will be a ssh connection to host_B with user name user_on_C, port number port_F;

b) On host_B, once it received the ssh request from port_F, it will forward it to port 22 of host_C, with user name user_on_C.


These two steps sounds confused (actually they are). And they only works under two pre-conditions:

a) port_F should be allowed to access (on some cloud server, only port 22 is opened by default) on host_B (you could use telnet command to check if port_F is open by command: `telnet $host_B $port_F` from host_A, it should return `Trying $host_B... Connected to $host_B ($host_B).` if port_F is open);

b) user_on_C should have password-less ssh key pair setup on host_B and host_C; (otherwise, from command line you will get timed out error when trying to connect and on vscode you will be prompt to input password for user_on_C on host_B).

**Solution 3 for Scenario 1**

In solution 2, There are still some pre-conditions to meet. Can we reduce this pre-conditions?

The answer is partially yes. The 'yes' part is that we could remove those two pre-conditions by introducing a local port forwarding on host_A. But this new solution could bring in new pre-conditions on host_A. Lets's see this new solution:

![alt text](/assets/img/2024-09-11-image-5.png)

Comparing to solution 2, we add an extra local port forwarding rule to host_A, which is 

```bash
# Corresponding to LPF_on_A in above image
# port_F1 is a local port number on host_A, it could be any unused port number.
ssh -L $port_F1:$host_B:$port_F user_on_B@host_B
```

So we can access host_C by this command on host_A:

```bash
# just mark it as cmd_1
ssh -p $port_F1 $user_on_C@localhost
```

On macOS, `cmd_1` could work (on macOS's own terminal.app) if you finish all required setup of local port forwarding on both host_A and host_B. But it won't work in vscode. 

On vscode, it will try to find the authorized keys for `user_on_C` on that macOS from `/Users/$user_on_C/.ssh/authorized_keys` when initiating accessing to `$user_on_C@localhost`. So the solution is:

Create a this new user directory `/Users/$user_on_C` on macOS, and copy all files from `~/.ssh` to it. The directory creating require `sudo` and you will find that this new directory's owner is `root` and its group is `admin`. And you as a genuine login user actually also belongs to `admin` group. So you can add access permission for admin group.

The commands for above steps are:

```bash
sudo mkdir /Users/$user_on_C
sudo cp -r ~/.ssh /Users/$user_on_C/
sudo chmod -R g+rx /Users/$user_on_C
```
So that you can access the files in this directory without `sudo` required. And this is the key for vscode to work properly.

Now `cmd_1` should work in vscode and make you connecting to host_C successfully.

There is another scenario I want to talk about.

**Scenario 2**

We have three machines:
* host_A is our laptop (we always work in starbucks); 
* host_B is a remote server with public IP address; 
* host_C is the GPU server at home or in office. 

Only host_A and host_C could access host_B individually. No other direct connections available.

Our goal is still to let vscode on host_A connect to host_C via remote ssh connection. 

![alt text](/assets/img/2024-09-11-image-6.png)

**Solution for Scenario 2**

This situation looks like impossible. host_B seems not be the intermidiate between A and C. But we can use a trick to make it possible. We can let host_C connect to host_B and monitor host_B's network, once host_B received trafics from specific port, it must forward them to host_C. 

This actually is another kind of port forwarding, called 'remote port forwarding'

On host_C, we can use the following command to set up a remote port forwarding:

```bash
# -R: remote port forwarding
ssh -fCN -R $port_F2:localhost:22 $user_on_B@$host_B
```

It means, there is a port forwarding rule created on host_C, this rule let any traffic coming to port_F2 of host_B will be forwarded to port 22 of host_C (represented as `localhost` in above command)

This remote port forwarding rule make connection from host_B to host_C possible. 

The left part of the solution is similar to Scenario 1 (host_A -> host_B --> host_C is available). We just need add a local port forwarding on host_A:

```bash
ssh -fCN -L $port_F1:localhost:$port_F2  $user_on_B@$host_B
```

Now we could connect to host_C via vscode on host_A by:

```bash
ssh -p $port_F1 $user_on_C@localhost
```

There is also issue about user_on_C should have the passwordless ssh key pair on host_A. The solution is the same as in solution 3 of scenario 1.