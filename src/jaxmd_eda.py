"""
JAX-MD Energy Decomposition Analysis (EDA) utilities.

This module provides the essential functions for creating EDA-enabled
SO3LR potentials that can decompose energy into components.
"""

import jax
import jax.numpy as jnp
from typing import Optional
from mlff.mdx.potential.mlff_potential_sparse import MLFFPotentialSparse
import pathlib

def so3lr_potential_eda(model_path: Optional[str] = None, dtype=jnp.float64):
    """
    Create a SO3LR potential with EDA components enabled.

    This creates an MLFF potential that outputs intermediate quantities
    necessary for energy decomposition analysis.

    Args:
        model_path: Path to SO3LR model parameters. If None, uses default path.
        dtype: JAX data type for the potential (jnp.float32 or jnp.float64).

    Returns:
        MLFF potential configured for EDA output
    """


    if model_path is None:
        # Use default SO3LR params directory
        package_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
        workdir_path = package_dir / 'so3lr' / 'so3lr' / 'params'
    else:
        workdir_path = pathlib.Path(model_path)

    # Create MLFF potential with energy components enabled
    potential = MLFFPotentialSparse.create_from_ckpt_dir(
        ckpt_dir=workdir_path,
        from_file=True,
        long_range_kwargs=dict(
            cutoff_lr=12.0,
            dispersion_energy_cutoff_lr_damping=2.0,
            neighborlist_format_lr='ordered_sparse',
        ),
        dtype=dtype,
        # Enable energy component output
        output_intermediate_quantities=[
            'mlff_atomic_energy',
            'zbl_repulsion',
            'electrostatic_energy',
            'dispersion_energy'
        ]
    )

    return potential