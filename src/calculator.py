"""
SO3LR-SF Calculator Module

This module provides the core calculator interface for energy calculations using SO3LR.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union
from ase import Atoms
from mlff.md.calculator_sparse import mlffCalculatorSparse

from .utils import read_structure, validate_structure
from .config import get_default_model_path


class So3lrSfCalculator:
    """
    A comprehensive SO3LR-SF energy calculator for molecular systems.

    This class provides a robust interface for molecular energy calculations with
    automatic model detection. It focuses on single energy calculations per structure
    to avoid redundant computations.

    Attributes:
        model_path (str): Path to SO3LR model parameters
        lr_cutoff (float): Long-range interaction cutoff distance
        dtype (type): Numerical precision for calculations
        output_per_atom_energy_components (bool): Whether per-atom components are enabled
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        lr_cutoff: float = 12.0,
        dispersion_energy_lr_cutoff_damping: float = 2.0,
        dtype: type = np.float32,
        output_per_atom_energy_components: bool = False
    ):
        """
        Initialize the SO3LR-SF energy calculator.

        Args:
            model_path: Path to SO3LR model parameters. If None, auto-detects.
            lr_cutoff: Long-range cutoff distance in Angstroms (default: 12.0)
            dispersion_energy_lr_cutoff_damping: Dispersion energy cutoff damping factor (default: 2.0)
            dtype: Numerical precision - np.float32 for speed, np.float64 for accuracy
            output_per_atom_energy_components: Enable per-atom energy decomposition

        Raises:
            FileNotFoundError: If model_path is None and automatic detection fails

        Example:
            >>> # Basic calculator with auto-detection
            >>> calc = So3lrSfCalculator()
            >>>
            >>> # High-precision calculator with per-atom components
            >>> calc = So3lrSfCalculator(dtype=np.float64,
            ...                         output_per_atom_energy_components=True)
        """
        # Auto-detect model path if not provided
        if model_path is None:
            model_path = get_default_model_path()

        self.model_path = model_path
        self.lr_cutoff = lr_cutoff
        self.dispersion_energy_lr_cutoff_damping = dispersion_energy_lr_cutoff_damping
        self.dtype = dtype
        self.output_per_atom_energy_components = output_per_atom_energy_components

        # Initialize calculator
        self._calculator = None
        self._init_calculator()

    def _init_calculator(self) -> None:
        """
        Initialize the underlying MLFFCalculator with SO3LR parameters.

        This method sets up the core calculator with the specified parameters
        and validates that the model can be loaded successfully.

        Raises:
            RuntimeError: If calculator initialization fails
        """
        try:
            import logging

            # Temporarily suppress JAX/checkpoint/MLFF logging
            loggers_to_suppress = [
                logging.getLogger('jax'),
                # logging.getLogger('MLFF'),
                logging.getLogger('orbax'),
                logging.getLogger('checkpoint'),
                logging.getLogger('so3lr'),
                logging.getLogger('jax._src'),
                logging.getLogger('jax._src.cache_key'),
                logging.getLogger('jax._src.compiler'),
                logging.getLogger('jax._src.xla_bridge'),
                logging.getLogger('absl')
            ]

            original_levels = {}
            for logger in loggers_to_suppress:
                original_levels[logger] = logger.level
                logger.setLevel(logging.CRITICAL)  # Even more restrictive

            try:
                self._calculator = mlffCalculatorSparse.create_from_ckpt_dir(
                    ckpt_dir=self.model_path,
                    lr_cutoff=self.lr_cutoff,
                    dispersion_energy_lr_cutoff_damping=self.dispersion_energy_lr_cutoff_damping,
                    from_file=False,
                    calculate_stress=False,  # We don't need stress calculations
                    dtype=self.dtype,
                    output_per_atom_energy_components=self.output_per_atom_energy_components
                )
            finally:
                # Restore original logging levels
                for logger, level in original_levels.items():
                    logger.setLevel(level)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SO3LR calculator: {e}")

    def calculate_energy(self, atoms: Union[Atoms, str, Path]) -> float:
        """
        Calculate the potential energy of a molecular system.

        This method computes the total potential energy using the SO3LR model,
        including all long-range interactions and corrections. Each call performs
        a complete energy calculation.

        Args:
            atoms: Molecular structure as ASE Atoms object or path to structure file
                  Supported formats: .xyz, .pdb, .sdf, and others supported by ASE

        Returns:
            float: Total potential energy in eV

        Raises:
            ValueError: If structure cannot be loaded or is invalid
            RuntimeError: If energy calculation fails

        Example:
            >>> calc = So3lrSfCalculator()
            >>> energy = calc.calculate_energy("molecule.xyz")
            >>> print(f"Potential energy: {energy:.3f} eV")
        """
        if isinstance(atoms, (str, Path)):
            atoms = read_structure(atoms)

        validate_structure(atoms)

        # Re-initialize calculator for fresh calculation
        self._init_calculator()

        # Set calculator and compute energy
        atoms.calc = self._calculator
        try:
            energy = atoms.get_potential_energy()
            return float(energy)
        except Exception as e:
            raise RuntimeError(f"Energy calculation failed: {e}")

    def get_per_atom_energy_components(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get per-atom energy components from the last calculation.

        This method retrieves detailed per-atom energy breakdown from the most recent
        energy calculation. Must be called after calculate_energy() and only works
        if output_per_atom_energy_components was enabled during initialization.

        Returns:
            dict or None: Dictionary mapping component names to per-atom energy arrays.
                         Keys typically include 'mlff_atomic_energy', 'zbl_repulsion',
                         'electrostatic_energy', 'dispersion_energy'. Returns None if
                         per-atom components were not enabled or no calculation performed.

        Raises:
            ValueError: If per-atom energy components were not enabled during initialization

        Example:
            >>> calc = So3lrSfCalculator(output_per_atom_energy_components=True)
            >>> energy = calc.calculate_energy("molecule.xyz")
            >>> components = calc.get_per_atom_energy_components()
            >>> if components:
            ...     for comp_name, values in components.items():
            ...         print(f"{comp_name}: total = {np.sum(values):.3f} eV")
        """
        if not self.output_per_atom_energy_components:
            raise ValueError(
                "Per-atom energy components not enabled. "
                "Initialize calculator with output_per_atom_energy_components=True"
            )

        if hasattr(self._calculator, 'get_per_atom_energy_components'):
            return self._calculator.get_per_atom_energy_components()
        else:
            return None