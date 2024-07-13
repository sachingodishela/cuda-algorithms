# cuda-algorithms
[![CI](https://github.com/sachingodishela/cuda-algorithms/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sachingodishela/cuda-algorithms/actions/workflows/ci.yml)


GPU utilized parallel algorithms for numerical methods and scientific computing. Each root level folder in this repository contains an algorithm and few examples using that algorithm.

## Setup Instructions
You can use any of the 2 methods to setup the repository on your linux machine:
### 1. Standard Method
1. Install latest [Nvidia drivers](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#driver-installation). To make sure it's installed, run this command:
``` sh
nvidia-smi
```
2. Install [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). (Recommended version 11.8). To make sure its install, run this command:
``` sh
nvcc --version
```
3. Install [VS Code](https://code.visualstudio.com/download) (Recommended code editor). 
4. For VS Code intellisense, add cuda toolkit's include path in `CPLUS_INCLUE_PATH` variable in your `/etc/profile` file.
5. Checkout this repository and open in your code editor:
``` sh
git clone git@github.com:sachingodishela/cuda-algorithms.git
cd cuda-algorithms
code .
``` 
### 2. Using Dev Container
1. Install [Docker](https://docs.docker.com/engine/install/), do not skip the [post-installation](https://docs.docker.com/engine/install/linux-postinstall/) steps. To test your installation, run this without sudo:
```
 docker run hello-world
```
3. Install latest [Nvidia drivers](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#driver-installation). To make sure it's installed, run this command:
``` sh
nvidia-smi
```
3. [Install](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation) and [configure](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuration) the Nvidia Container Toolkit. Don't skip the [Rootless mode](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuration) while configuring. To test your configuration, run this without sudo:
```
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```
4. Install [VS Code](https://code.visualstudio.com/download) and install teh [Dev Container](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.
4. Checkout this repository and open in Visual Studio Code:
``` sh
git clone git@github.com:sachingodishela/cuda-algorithms.git
cd cuda-algorithms
code .
```
5. When you get a prompt to open the workspace in dev container, click yes.

## Run/Debug Instructions
If using VS Code:
1. Open any `.cu` file and press `Ctrl + Shift + B` to run the opened file.
2. Open any `.cu` file and press `F5` to debug the opened file.

## Contribution
Just raise a PR and I'll take a look. Keep these in mind while developing:
1. Your algorithm must be auto-scalable to use as many GPUs as avaiable.
2. Don't forget to update the root level README with Changelog.
3. Add a folder level README if you're adding a new algorithm folder to the repo.

## Changelog
|Date|Time|Author|Change|
|-|-|-|-|
13-July-2024|18:25|[@sachingodishela](https://github.com/sachingodishela)|Vector Addition with CPU/GPU execution times comparision.|
12-July-2024|20:09|[@sachingodishela](https://github.com/sachingodishela)|Dev Container Support & Run hello-cuda in CI.|
11-July-2024|18:27|[@sachingodishela](https://github.com/sachingodishela)|Update README with setup and build instructions.|
11-July-2024|18:08|[@sachingodishela](https://github.com/sachingodishela)|Added vscode configs for building, running and debugging the files in linux environment. Added first program "hello-cuda" which prints avaiable GPUs and their properties.|
11-July-2024|16:37|[@sachingodishela](https://github.com/sachingodishela)|Created this repository.|
