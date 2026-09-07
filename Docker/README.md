# SO3LR-SF Docker Image

Containerized build of SO3LR-SF with the `so3lrsf` CLI and a pinned environment
(JAX 0.5.3, CUDA 12, commit-pinned `so3lr`/`mlff`). Runs CPU-only anywhere, or
on GPU with the NVIDIA Container Toolkit.

## Run

```bash
# Pull the pinned image
docker pull hamzaibrahim21/so3lr-sf:v0.1.0

# Interactive shell
docker run --rm -it hamzaibrahim21/so3lr-sf:v0.1.0

# Run a calculation (mount your input/output dir, and -w into it)
docker run --rm -v "$PWD/data:/data" -w /data \
  hamzaibrahim21/so3lr-sf:v0.1.0 \
  so3lrsf --protein protein.pdb --ligands ligands.sdf --verbose

# On GPU: add --gpus all
docker run --gpus all --rm -v "$PWD/data:/data" -w /data \
  hamzaibrahim21/so3lr-sf:v0.1.0 \
  so3lrsf --protein protein.pdb --ligands ligands.sdf --verbose
```

See all options with `so3lrsf --help`.

## Rebuild

Build from the repository root (not the `Docker/` dir), so `pyproject.toml`,
`uv.lock`, and `src/` are in the build context:

```bash
docker build -f Docker/Dockerfile -t hamzaibrahim21/so3lr-sf:<tag-name> .
docker push hamzaibrahim21/so3lr-sf:<tag-name>
```

The build installs from `uv.lock` with `uv sync --frozen`, so dependency
versions — including the commit-pinned `so3lr` and `mlff` — are reproduced
exactly.
