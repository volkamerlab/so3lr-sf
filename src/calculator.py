"""
SO3LR-SF Calculator Module

This module provides the core calculator interface for energy calculations using SO3LR.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

# Try to import JAX-MD components for enhanced performance
try:
    import jax
    import jax.numpy as jnp
    from jax_md import space
    import so3lr
    from so3lr import to_jax_md, So3lrCalculator
    from so3lr.jaxmd_utils import neighbor_list_featurizer
    from mlff.mdx.potential import MLFFPotentialSparse
    _jax_available = True
except ImportError:
    _jax_available = False

from .utils import validate_structure
from .molecule_loader import load_ase_structure

logger = logging.getLogger(__name__)

# Long-range cutoff (Angstroms) for dispersion and the long-range neighbour list.
DISPERSION_LR_CUTOFF = 1000.0

class So3lrSfCalculator:
    """
    A comprehensive SO3LR-SF energy calculator for molecular systems.

    This class provides a robust interface for molecular energy calculations with
    automatic model detection. It focuses on single energy calculations per structure
    to avoid redundant computations.

    Attributes:
        elec_lr_cutoff (float): Electrostatic long-range cutoff distance
        dtype (type): Numerical precision for calculations
        dp (bool): Double precision flag
        output_per_atom_energy_components (bool): Whether per-atom components are enabled
    """

    def __init__(
        self,
        dispersion_energy_lr_cutoff_damping: float = 2.0,
        dtype: type = np.float32,
        output_per_atom_energy_components: bool = False,
        dp: bool = False,
        elec_lr_cutoff: float = 10.0
    ):
        """
        Initialize the SO3LR-SF energy calculator.

        The SO3LR model parameters are always loaded from the bundled `so3lr`
        package, so no model path is required.

        Args:
            dispersion_energy_lr_cutoff_damping: Dispersion energy cutoff damping factor (default: 2.0)
            dtype: Numerical precision - np.float32 for speed, np.float64 for accuracy
            output_per_atom_energy_components: Enable per-atom energy decomposition
            dp: Enable double precision (float64). Overrides dtype when True.
            elec_lr_cutoff: Long-range cutoff (Angstroms) for the electrostatic term
                (default: 10.0). Dispersion and the long-range neighbour list stay pinned at
                DISPERSION_LR_CUTOFF (1000 A). Recommended: 10 for ranking / relative potency,
                1000 for absolute interaction energies compared to DFT.

        Example:
            >>> # Basic calculator (electrostatics cut at 10 A, dispersion at 1000 A)
            >>> calc = So3lrSfCalculator()
            >>>
            >>> # Calculator with EDA
            >>> calc = So3lrSfCalculator(output_per_atom_energy_components=True)
            >>>
            >>> # Electrostatics cut at 10 A, dispersion still at 1000 A
            >>> calc = So3lrSfCalculator(elec_lr_cutoff=10.0)
        """
        # Handle double precision flag
        if dp:
            self.dtype = np.float64
            # Enable JAX double precision if JAX is available
            if _jax_available:
                jax.config.update("jax_enable_x64", True)
        else:
            self.dtype = dtype
            # Disable JAX double precision if explicitly not requested and JAX is available
            if _jax_available and dtype != np.float64:
                jax.config.update("jax_enable_x64", False)

        self.dispersion_energy_lr_cutoff_damping = dispersion_energy_lr_cutoff_damping
        self.elec_lr_cutoff = elec_lr_cutoff
        self.dp = dp
        self.output_per_atom_energy_components = output_per_atom_energy_components

        # Initialize calculator (always try JAX-MD first, fallback to MLFF)
        self._calculator = None
        self._jax_setup = None
        self._last_aux_data = {}
        self._init_calculator()

    def _init_calculator(self) -> None:
        """
        Initialize calculator. JAX-MD is tried first, falls back to MLFF on failure.
        """
        # Don't initialize anything here - JAX-MD will be tried first in calculate_energy
        # and MLFF will be initialized only when needed as fallback
        self._calculator = None

    def _init_so3lr_calculator(self) -> None:
        """Initialize the SO3LR calculator."""
        try:
            logger.debug("Initializing SO3LR calculator (ASE So3lrCalculator fallback)")
            logger.debug(f"Calculator parameters: lr_cutoff={DISPERSION_LR_CUTOFF}, "
                        f"elec_lr_cutoff={self.elec_lr_cutoff}, "
                        f"dispersion_damping={self.dispersion_energy_lr_cutoff_damping}, "
                        f"output_per_atom={self.output_per_atom_energy_components}")

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
            for log in loggers_to_suppress:
                original_levels[log] = log.level
                log.setLevel(logging.CRITICAL)

            try:
                self._calculator = So3lrCalculator(
                    lr_cutoff=DISPERSION_LR_CUTOFF,
                    dispersion_energy_cutoff_lr_damping=self.dispersion_energy_lr_cutoff_damping,
                    calculate_stress=False,
                    dtype=self.dtype,
                    add_energy_shift=False,
                    output_per_atom_energy_components=self.output_per_atom_energy_components
                )
            finally:
                # Restore original logging levels
                for log, level in original_levels.items():
                    log.setLevel(level)

            logger.debug("SO3LR calculator initialized successfully")
        except Exception as e:
            logger.debug(f"SO3LR calculator initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize SO3LR calculator: {e}")

    def calculate_energy(self, atoms: Union[Atoms, str, Path]) -> float:
        """
        Calculate the potential energy of a molecular system.

        This method computes the total potential energy using either JAX-MD mode
        for enhanced performance or SO3LR mode for full feature support.

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
            logger.debug(f"Loading structure from file: {atoms}")
            atoms = load_ase_structure(atoms)[0]

        validate_structure(atoms)

        if _jax_available:
            try:
                return self._calculate_energy_jax_md(atoms)
            except Exception as e:
                logger.debug(f"JAX-MD calculation failed ({e}), falling back to MLFF calculator")
                # Fallback to MLFF calculator
                return self._calculate_energy_so3lr(atoms)
        else:
            logger.debug("JAX not available, using MLFF calculator")
            return self._calculate_energy_so3lr(atoms)

    @staticmethod
    def _residue_charge_info(atoms: Atoms):
        """
        Build per-fragment charge info for a separated dimer.

        mlff conserves charge per fragment when the graph carries `residue_charge` /
        `residue_segments`, instead of spreading one total charge over every atom. The
        last entry of `residue_charge` is reserved for padding nodes, matching what
        mlff's own dataloader emits; the jax-md graph has no padding nodes, so nothing
        is ever assigned to it here.

        Only a structure tagged 'dimer_translated' is split. Anything else returns
        (None, None) and falls back to global charge conservation.

        Returns:
            Tuple of (residue_charge, residue_segments) or (None, None)
        """
        if atoms.info.get('structure_type') != 'dimer_translated':
            return None, None

        charge_a = atoms.info.get('charge_a')
        charge_b = atoms.info.get('charge_b')
        num_a = atoms.info.get('num_a')
        num_b = atoms.info.get('num_b')
        if any(v is None for v in (charge_a, charge_b, num_a, num_b)):
            logger.warning("structure_type='dimer_translated' but charge_a/charge_b/num_a/num_b "
                           "are incomplete; falling back to global charge conservation")
            return None, None

        num_a, num_b = int(num_a), int(num_b)
        if num_a + num_b != len(atoms):
            logger.warning(f"Fragment sizes ({num_a} + {num_b}) do not match the structure "
                           f"({len(atoms)} atoms); falling back to global charge conservation")
            return None, None
        if num_a == 0 or num_b == 0:
            logger.warning("One fragment is empty; falling back to global charge conservation")
            return None, None

        charge_a, charge_b = int(charge_a), int(charge_b)
        total_charge = atoms.info.get('charge')
        if total_charge is not None and charge_a + charge_b != int(total_charge):
            logger.warning(f"Fragment charges ({charge_a} + {charge_b}) do not sum to the total "
                           f"charge ({int(total_charge)}); using the fragment charges")

        residue_charge = np.array([charge_a, charge_b, 0])
        residue_segments = np.concatenate([np.repeat(0, num_a), np.repeat(1, num_b)])
        logger.debug(f"Per-fragment charges: A={charge_a} ({num_a} atoms), "
                     f"B={charge_b} ({num_b} atoms)")
        return residue_charge, residue_segments

    def _build_potential(self, jax_dtype):
        """Build the bundled SO3LR potential with the electrostatic long-range
        cutoff decoupled from the dispersion / neighbour-list cutoff.
        """
        long_range_kwargs = dict(
            cutoff_lr=DISPERSION_LR_CUTOFF,
            dispersion_energy_cutoff_lr_damping=self.dispersion_energy_lr_cutoff_damping,
            neighborlist_format_lr='ordered_sparse',
            coulomb_kspace_do_ewald=False,
            coulomb_kspace_interp_nodes=4,
            electrostatic_cutoff_lr=self.elec_lr_cutoff,
        )
        extra_kwargs = {}
        if self.output_per_atom_energy_components:
            extra_kwargs['output_intermediate_quantities'] = [
                'nn_energy', 'zbl_repulsion', 'electrostatic_energy', 'dispersion_energy'
            ]
        return MLFFPotentialSparse.create_from_workdir(
            workdir=Path(so3lr.__file__).parent / 'params',
            from_file=True,
            dtype=jax_dtype,
            long_range_kwargs=long_range_kwargs,
            **extra_kwargs,
        )

    def _calculate_energy_jax_md(self, atoms: Atoms) -> float:
        """Calculate energy using JAX-MD for enhanced performance."""
        # logger.debug("Using JAX-MD mode for energy calculation")

        # Setup JAX-MD system with appropriate precision
        jax_dtype = jnp.float64 if self.dp or self.dtype == np.float64 else jnp.float32
        positions = jnp.array(atoms.get_positions(), dtype=jax_dtype)
        atomic_numbers = jnp.array(atoms.get_atomic_numbers(), dtype=jnp.int32)

        # Total charge / spin come from the ASE atoms.info dict (set upstream from
        # the CLI --charge-* args). so3lr's stock JAX-MD featurizer hardcodes
        # total_charge=0, so we read them here and inject them below.
        total_charge = float(atoms.info.get('charge', 0.0))
        multiplicity = atoms.info.get('multiplicity')
        num_unpaired_electrons = float(multiplicity - 1) if multiplicity is not None else 0.0
        logger.debug(f"JAX-MD total_charge={total_charge}, num_unpaired_electrons={num_unpaired_electrons}")

        residue_charge, residue_segments = self._residue_charge_info(atoms)

        # Setup displacement function for free boundary conditions
        displacement, shift = space.free()

        potential = self._build_potential(jax_dtype)


        # Neighbor lists are charge-independent, so reuse so3lr's to_jax_md for
        # them and discard its charge-blind energy_fn.
        neighbor_fn, neighbor_fn_lr, _ = to_jax_md(
            potential=potential,
            displacement_or_metric=displacement,
            box_size=None,
            species=atomic_numbers,
            capacity_multiplier=1.25,
            buffer_size_multiplier_sr=1.25,
            buffer_size_multiplier_lr=1.25,
            minimum_cell_size_multiplier_sr=1.0,
            disable_cell_list=True,
            fractional_coordinates=False
        )

        # Rebuild the energy_fn with the same featurizer so3lr uses, but stamp the
        # real total_charge / num_unpaired_electrons onto the graph so the charge
        # actually reaches the model (so3lr's featurizer otherwise hardcodes 0).
        featurizer = neighbor_list_featurizer(
            displacement, atomic_numbers, fractional_coordinates=False
        )
        total_charge_arr = jnp.asarray([total_charge], dtype=jax_dtype)
        num_unpaired_arr = jnp.asarray([num_unpaired_electrons], dtype=jax_dtype)
        residue_charge_arr = (
            jnp.asarray(residue_charge, dtype=jax_dtype) if residue_charge is not None else None
        )
        residue_segments_arr = (
            jnp.asarray(residue_segments, dtype=jnp.int32) if residue_segments is not None else None
        )

        def energy_fn(R, neighbor, neighbor_lr, has_aux=False, **energy_fn_kwargs):
            graph = featurizer(R, neighbor, neighbor_lr, **energy_fn_kwargs)
            graph = graph._replace(
                total_charge=total_charge_arr,
                num_unpaired_electrons=num_unpaired_arr,
                residue_charge=residue_charge_arr,
                residue_segments=residue_segments_arr,
            )
            if has_aux:
                return potential(graph, has_aux=True)
            return potential(graph).sum()

        # Initialize neighbor lists
        nbrs = neighbor_fn.allocate(positions, box=None)
        nbrs_lr = neighbor_fn_lr.allocate(positions, box=None)

        # Calculate energy
        if self.output_per_atom_energy_components:
            # For EDA-enabled potential, we need to handle aux output
            result = energy_fn(positions, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=None, has_aux=True)
            if isinstance(result, tuple):
                energy = result[0]
                self._last_aux_data = result[1]  # Store aux data for component extraction
            else:
                energy = result
                self._last_aux_data = {}
        else:
            energy = energy_fn(positions, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=None)

        # Handle different energy result formats
        energy_array = np.array(energy)
        # logger.debug(f"Energy result shape: {energy_array.shape}, dtype: {energy_array.dtype}, value: {energy_array}")

        if energy_array.ndim == 0:
            energy_value = float(energy_array)
        elif energy_array.ndim == 1 and len(energy_array) == 1:
            energy_value = float(energy_array[0])
        else:
            energy_value = float(np.sum(energy_array))

        # logger.debug(f"JAX-MD energy calculation completed: {energy_value:.6f} eV")
        return energy_value



    def _calculate_energy_so3lr(self, atoms: Atoms) -> float:
        """Calculate energy using standard SO3LR calculator."""
        logger.debug("Using SO3LR mode for energy calculation")

        # Initialize MLFF calculator for force/optimization support
        self._init_so3lr_calculator()

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
                         Keys typically include 'nn_energy', 'zbl_repulsion',
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
            logger.debug("Per-atom energy components not enabled during initialization")
            raise ValueError(
                "Per-atom energy components not enabled. "
                "Initialize calculator with output_per_atom_energy_components=True"
            )

        logger.debug("Retrieving per-atom energy components from calculator")

        # Try JAX-MD aux data first
        if hasattr(self, '_last_aux_data') and self._last_aux_data:
            components = {}
            component_keys = ['nn_energy', 'zbl_repulsion', 'electrostatic_energy', 'dispersion_energy']

            for component_key in component_keys:
                if component_key in self._last_aux_data:
                    components[component_key] = np.array(self._last_aux_data[component_key])
                else:
                    logger.debug(f"Component {component_key} not found in aux data")

            return components if components else None

        # Fall back to MLFF calculator
        if hasattr(self._calculator, 'get_per_atom_energy_components'):
            return self._calculator.get_per_atom_energy_components()
        else:
            logger.debug("Calculator does not support per-atom energy components")
            return None