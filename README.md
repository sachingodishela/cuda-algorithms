# cuda-algorithms
[![CI](https://github.com/sachingodishela/cuda-algorithms/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sachingodishela/cuda-algorithms/actions/workflows/ci.yml)


GPU utilized parallel algorithms for numerical methods and scientific computing.

## How to run
1. Checkout this repository and open in Visual Studio Code:
``` sh
git clone git@github.com:sachingodishela/cuda-algorithms.git
cd cuda-algorithms
code .
``` 

2. Open any `.cu` file and press `Ctrl + Shift + B` to run the opened file.
3. Open any `.cu` file and press `F5` to debug the opened file.

## Contribution
Just raise a PR and I'll take a look. Keep these in mind while developing:
1. Your algorithm must be auto-scalable to use as many GPUs as avaiable.
2. Don't forget to update the root level README with Changelog.
3. Add a folder level README if you're adding a new algorithm folder to the repo.

## Changelog
|Date|Time|Author|Change|
|-|-|-|-|
11-July-2024|18:27|[@sachingodishela](https://github.com/sachingodishela)|Update README with setup and build instructions.|
11-July-2024|18:08|[@sachingodishela](https://github.com/sachingodishela)|Added vscode configs for building, running and debugging the files in linux environment. Added first program "hello-cuda" which prints avaiable GPUs and their properties.|
11-July-2024|16:37|[@sachingodishela](https://github.com/sachingodishela)|Created this repository.|
