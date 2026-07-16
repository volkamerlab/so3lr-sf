"""
JAX-MD Energy Decomposition Analysis (EDA) utilities.

This module provides the essential functions for creating EDA-enabled
SO3LR potentials that can decompose energy into components.
"""

import jax.numpy as jnp
from so3lr import So3lrPotential

def so3lr_potential_eda(dtype=jnp.float64,
                       cutoff_lr: float = 12.0,
                       dispersion_energy_cutoff_lr_damping: float = 2.0):
    """
    Create a SO3LR potential with EDA components enabled.

    This wraps the supported ``So3lrPotential`` factory and additionally
    requests the intermediate quantities necessary for energy
    decomposition analysis.

    Args:
        dtype: JAX data type for the potential (jnp.float32 or jnp.float64).
        cutoff_lr: Long-range cutoff distance in Angstroms (default: 12.0).
        dispersion_energy_cutoff_lr_damping: Dispersion energy cutoff damping factor (default: 2.0).

    Returns:
        MLFF potential configured for EDA output
    """

    # Use the supported So3lrPotential factory, forwarding only the
    # extra keyword needed to expose per-component energies for EDA.
    return So3lrPotential(
        lr_cutoff=cutoff_lr,
        dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
        dtype=dtype,
        # Enable energy component output for EDA
        output_intermediate_quantities=[
            'nn_energy',
            'zbl_repulsion',
            'electrostatic_energy',
            'dispersion_energy'
        ]
    )