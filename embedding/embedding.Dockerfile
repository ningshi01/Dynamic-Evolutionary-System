# Use an official NVIDIA PyTorch image as a base for maximum compatibility.
# The 23.10-py3 tag includes CUDA 12.2 and PyTorch 2.1, a stable combination.
# Using a mirror for better accessibility.

# Attention! The address may be wrong !!!
# FROM docker.1ms.run/nvidia/pytorch:23.10-py3

FROM nvcr.io/nvidia/pytorch:23.10-py3

# Install a specific, stable version of vLLM using pip.

# This ensures that vLLM is compiled against the libraries in this base image.
RUN pip install vllm==0.4.2

# Expose the default port for the OpenAI-compatible server.
EXPOSE 8000 