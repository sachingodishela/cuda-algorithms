FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y git

# ENV PATH="/usr/local/cuda/bin:${PATH}"
# ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
# ENV CUDA_HOME="/usr/local/cuda"
ENV CPLUS_INCLUDE_PATH="/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}"

# Add your additional commands here

CMD ["/bin/bash"]