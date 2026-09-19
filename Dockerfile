# Reproducible build/benchmark environment for AegisQ-HPC (CPU + MPI).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        ca-certificates \
        libopenmpi-dev \
        openmpi-bin \
        libomp-dev \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/aegisq
COPY pyproject.toml README.md LICENSE ./
COPY aegisq ./aegisq
COPY cpp ./cpp
COPY CMakeLists.txt Makefile ./
COPY scripts ./scripts
COPY tests ./tests
COPY examples ./examples

RUN python3 -m pip install --no-cache-dir \
        numpy pytest pybind11 cryptography pandas matplotlib liboqs-python qiskit \
    && ./scripts/build.sh \
    && python3 -m pip install --no-cache-dir -e .

# OpenMPI refuses to run as root without this opt-in.
ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

CMD ["aegisq", "doctor"]
