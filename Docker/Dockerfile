# syntax=docker/dockerfile:1

# NVIDIA CUDA 12 base image, compatible with JAX 0.5.3.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # uv configuration
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# System dependencies. git is required for the git-sourced deps (so3lr, mlff);
# build-essential/curl support building any sdist-only wheels; the libX*/libGL/
# libglib runtime libs are required by rdkit's drawing module (pulled in by prolif).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    libxrender1 \
    libxext6 \
    libsm6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv from its official image (static binary, pinned to the version
# used to generate uv.lock).
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /usr/local/bin/

WORKDIR /app

# 1) Resolve & install dependencies first for better layer caching.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable --python 3.12

# 2) Copy the project source and install the project itself.
COPY so3lr_sf.py pytest.ini ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --python 3.12

# 3) Add the JAX CUDA 12 plugin so JAX can use the GPU. The base pin in
#    pyproject.toml is CPU-only jaxlib; this layers the CUDA PJRT plugin on top.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "jax[cuda12]==0.5.3"

# Interactive shells start inside the project virtual environment.
RUN echo 'source /app/.venv/bin/activate' >> /root/.bashrc

RUN chmod 755 /root

# Default command
CMD ["/bin/bash"]
