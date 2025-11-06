# Use NVIDIA CUDA runtime base image compatible with JAX 0.5.3
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_CACHE_DIR=/opt/poetry-cache
ENV POETRY_VENV_IN_PROJECT=1
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install system dependencies (Python 3.12 is default in Ubuntu 24.04)
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python3.12 to python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN chmod +x $POETRY_HOME/bin/poetry

# Set working directory
WORKDIR /app

# Copy poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry and install dependencies only (handle externally managed environment)
RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry install --with test --no-interaction --no-ansi --no-root

# Copy the rest of the application
COPY . .

# Install the current project in development mode
RUN poetry install --only-root

# Install JAX with CUDA support (compatible with CUDA 12.6)
RUN .venv/bin/pip install --upgrade "jax[cuda12]"==0.5.3 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Download SO3LR model parameters
RUN mkdir -p so3lr && \
    cd so3lr && \
    curl -L https://github.com/general-molecular-simulations/so3lr/archive/main.tar.gz | tar -xz --strip-components=2 so3lr-main/so3lr/params

# Set up proper permissions
RUN chmod +x setup.py

# Expose port (if needed for any web services)
EXPOSE 8080

# Activate virtual environment by default
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Create activation script for interactive shells
RUN echo 'source /app/.venv/bin/activate' >> /root/.bashrc

# Default command
CMD ["/bin/bash"]
