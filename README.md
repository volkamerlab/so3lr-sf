# **SO3LR-SF** - Advancing computational drug discovery with machine learning force fields.
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/volkamerlab/so3lr-sf/graph/badge.svg?token=A0X4RNCKBI)](https://codecov.io/gh/volkamerlab/so3lr-sf)
[![CI](https://github.com/volkamerlab/so3lr-sf/workflows/CI/badge.svg)](https://github.com/volkamerlab/so3lr-sf/actions)

SO3LR-SF is a comprehensive Python package for calculating protein-ligand interaction energies using SO3LR machine learning force fields. It provides advanced features including structure optimization, explainability analysis, and molecular visualization.

<details>
<summary><h2>🚀 Features</h2></summary>

### Core Functionality
- **Protein-Ligand Binding Energy Calculation**: Calculate binding energies using SO3LR force fields
- **Protein Trimming**: Trim protein structures around ligands to reduce computational cost
- **Structure Optimization**: Optimize protein, ligand, and complex structures with many ASE optimization algorithms
- **Energy decomposition analysis (EDA)**: Analysis of each energy term individually
- **Ligand Explainability**: Generate per-atom energy contributions and 2D molecular heatmaps for ligands
- **Protein Explainability**: ProLIF-powered interaction fingerprinting with residue-level energy contributions and bond-colored visualizations
- **3D Energy Visualization**: PyMOL-based 3D visualization of protein energy components with side-by-side comparison and energy-based coloring

### File Format Support
- **Protein input**: PDB (preferred for optimization and trimming) and XYZ formats
- **Multi-molecule files**: Automatic splitting of multi-ligand SDF, XYZ or PDB files
- **Outputs**: JSON results, XYZ optimized structures, PNG heatmaps

### Workflow Tracking
- **Silent/Verbose Modes**: Adjustable output verbosity for production deployment and development
- **Comprehensive Error Recovery**: Automatic failure detection and detailed diagnostic reporting
- **Optimization logging**: Complete trajectory logging and convergence analysis for quality assurance

</details>

<details>
<summary><h2>📦 Installation</h2></summary>

### Prerequisites
- Python 3.12 or higher
- Poetry (package manager)
- curl (for downloading model parameters)

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/volkamerlab/so3lr-sf.git
cd so3lr-sf

# Run the setup script (installs dependencies and downloads SO3LR model parameters)
python setup.py
```

The setup script will:
1. Install all dependencies using Poetry
2. Download SO3LR model parameters from the official repository
3. Verify the installation

### Manual Installation (Alternative)

If you prefer manual installation:

```bash
# Clone the repository
git clone https://github.com/volkamerlab/so3lr-sf.git
cd so3lr-sf

# Install dependencies
poetry install --with test

# Download model parameters
mkdir -p so3lr
cd so3lr
curl -L https://github.com/general-molecular-simulations/so3lr/archive/main.tar.gz | tar -xz --strip-components=2 so3lr-main/so3lr/params
cd ..
```

### Verify Installation

```bash
# Run tests to verify everything works
poetry run pytest

# Or test with a simple calculation
poetry run python so3lr_sf.py --protein tests/test_data/alanine.xyz --ligands tests/test_data/water.sdf
```

</details>

<details>
<summary><h2>🏗️ Architecture</h2></summary>

### Module Organization
```
src/
├── calculator.py                      # SO3LR calculator implementation
├── config.py                          # Configuration management and model path discovery
├── interaction_energy.py              # Main energy calculation functions
├── optimization.py                    # Structure optimization through constraint or free optimization
├── constraint.py                      # Handle optimization constraints either by-atom or by-residue
├── trim.py                            # Trim protein structure either by-atom or by-residue
├── explainability.py                  # Energy-based explainability analysis and visualization
├── explain_utils.py                   # Utility functions for explainability analysis
├── molecule_loader.py                 # Molecule loading and file format handling
├── utils.py                           # General utility functions and I/O operations
├── visualizer.py                      # 3D Protein-Ligand Energy Visualizer for PyMOL
└── __init__.py                        # Package interface
```
### Full pipeline output directory structure
```
results_steps_{#_steps}_fmax{FMAX}/
├── individual_ligands/                     # Extracted ligands (for multi-SDF)
│   └── ligand_001.sdf
├── ligand_exp/                             # Explainability heatmaps
│   └── ligand_001_heatmap.png
├── pl_2d_exp/                              # 2D protein-ligand interaction heatmaps
│   └── ligand_001_protein_interaction.png
├── pl_3d_exp/                              # 3D protein-ligand interaction heatmaps
│   ├── complex_ligand_001_constrained_opt.pdb
│   └── ligand_001_3d_visualization.pml
├── constrained_opt_complexes/              # Optimized protein-ligand complexes
│   └── ligand_001_complex_constrained_opt.xyz
├── constrained_opt_components/             # Extracted optimized protein and ligand components
│   ├── {protein_name}_ligand_001_constrained_opt.xyz
│   └── ligand_001_constrained_opt.xyz
├── free_opt_ligands/                       # Free ligands for strain calculation (strain modes)
│   └── ligand_001_free_opt.xyz
├── free_opt_protein/                       # Free protein for strain calculation (strain-prot mode)
│   └── {protein_name}_free_opt.xyz
├── results_summary.json                    # Main results file
└── optimization_log.json                   # Optimization details
```

</details>


<details>
<summary><h2>📋 Command Line Arguments</h2></summary>

### Required Arguments
- `--protein`: Path to protein structure file
- `--ligands`: Path to ligand file, directory, or multi-SDF file

### Workflow Options
- `--trim FLOAT`: Trim protein around ligand(s) with specified radius in Angstroms
- `--optimize FLOAT`: Optimize structures with constrained radius around ligand (default: 4.0 Å)
- `--exp-lig`: Generate explainability analysis and heatmaps
- `--exp-prot`: Generate protein explainability with protein-ligand interaction analysis and interacting residue coloring depending on their energy contribution
- `--exp-3d`: Generate 3D PyMOL visualization of protein energy components

### Trimming Parameters
- `--trim-lig FILE`: Specific ligand file to use for trimming

### Optimization Parameters
- `--optimize FLOAT`: Enable optimization with specified radius around ligand in Angstroms (default: 4.0 if no value provided)
- `--optimization-mode {no-strain,strain,strain-prot}`: Strain energy calculation mode (default: no-strain)
  - **no-strain**: Traditional constrained optimization without strain energy corrections
  - **strain**: Add ligand strain energy (cost of ligand deformation from optimal free conformation)
  - **strain-prot**: Add both ligand and protein strain energies for complete binding thermodynamics
- `--optimizer {FIRE,FIRE2,LBFGS,BFGS,BFGSLineSearch,LBFGSLineSearch,GPMin,MDMin,ODE12r,GoodOldQuasiNewton,QuasiNewton}`: Optimization algorithm (default: FIRE)
- `--fmax FLOAT`: Force convergence criterion in eV/Å (default: 0.05)
- `--steps INT`: Maximum optimization steps (default: 100)

### Model Parameters
- `--model-path PATH`: Path to SO3LR model parameters (auto-detected if not specified)

### Charge Parameters
- `--charge-lig INT`: Charge of the ligand (default: 0)
- `--charge-prot INT`: Charge of the protein (default: 0)
- `--charge-cpx INT`: Charge of the complex (default: 0)

### Logging & Output
- `-v, --verbose`: Enable detailed logging output (INFO level)
- `--debug`: Enable debug level logging (includes verbose output and detailed debugging information)
- `--opt-log`: Save optimization details to JSON file

</details>


<details>
<summary><h2>🔧 Quick Start</h2></summary>

### Basic Usage

```bash
# Simple protein-ligand interaction calculation
python so3lr_sf.py --protein protein.pdb --ligands ligand.sdf

# With structure optimization
python so3lr_sf.py --protein protein.pdb --ligands ligand.sdf --optimize 4.0

# With constrained optimization + ligand-strain energy and 2D explainable ligand energies
python so3lr_sf.py --protein protein.pdb --ligands ligands.sdf \
  --trim 10.0 --optimize 4.0 --optimization-mode strain --exp-lig --verbose
```

### Python API

```python
from src import So3lrSfCalculator, protein_ligand_interaction

# Initialize calculator
calc = So3lrSfCalculator()

# Simple interaction energy
interaction_energy = protein_ligand_interaction(
    "protein.pdb", "ligand.sdf", calc
)
print(f"Interaction energy: {interaction_energy:.3f} eV")

# With explainability analysis
interaction_energy, analysis = protein_ligand_interaction(
    "protein.pdb", "ligand.sdf", calc,
    explainability=True,
    heatmap_output="heatmap.png"
)
print(f"Binding energy: {interaction_energy * 23.06:.1f} kcal/mol")
print(f"Component contributions: {analysis['component_totals']}")
```

</details>


<details>
<summary><h2>💡 Examples</h2></summary>

### Example 1: Basic Calculation
```bash
python so3lr_sf.py \
    --protein tests/test_data/alanine.xyz \
    --ligands tests/test_data/water.sdf
```

### Example 2: Multi-Ligand Screening + Trim the protein
```bash
python so3lr_sf.py \
    --protein target.pdb \
    --ligands ligand_library.sdf \
    --trim 10.0 \
    --trim-lig ref_lig.sdf \
    --verbose
```

### Example 3: Constrained Optimization (Default)
```bash
python so3lr_sf.py \
    --protein protein.pdb \
    --ligands ligands.sdf \
    --optimize 4.0 \
    --optimization-mode no-strain \
    --optimizer FIRE \
    --fmax 0.05 \
    --steps 100 \
    --verbose
```

### Example 4: Constrained Optimization + ligand-strain energy + protein-strain energy
```bash
python so3lr_sf.py \
    --protein protein.pdb \
    --ligands ligands.sdf \
    --optimize 4.0 \
    --optimization-mode strain-prot \
    --optimizer FIRE \
    --fmax 0.05 \
    --steps 100 \
    --verbose
```

### Example 5: Full Analysis Pipeline
```bash
python so3lr_sf.py \
    --protein protein.pdb \
    --ligands multi_ligands.sdf \
    --trim 8.0 \
    --optimize 4.0 \
    --optimization-mode strain \
    --exp-lig \
    --opt-log \
    --verbose
```

### Example 6: Protein-Ligand Interaction Analysis
```bash
python so3lr_sf.py \
    --protein protein.pdb \
    --ligands ligand.sdf \
    --exp-prot \
    --verbose
```

### Example 7: With Explicit Charge Parameters
```bash
python so3lr_sf.py \
    --protein protein.pdb \
    --ligands ligand.sdf \
    --charge-prot -2 \
    --charge-lig 1 \
    --charge-cpx -1 \
    --optimize 4.0 \
    --verbose
```

### Example 8: Debug Mode for Troubleshooting
```bash
python so3lr_sf.py \
    --protein protein.pdb \
    --ligands ligand.sdf \
    --debug \
    --optimize 4.0
```

</details>

<details>
<summary><h2>📊 Output Files</h2></summary>

### Results Summary example (`results_summary.json`)
```json
{
  "workflow_parameters": {
    "protein": "path/to/protein.xyz",
    "ligands_source": "path/to/ligand.xyz",
    "trim": null,
    "trim_radius": null,
    "optimize": 3.0,
    "optimization_mode": "strain-prot",
    "opt_radius": 3.0,
    "ligand_strain_calculation": true,
    "protein_strain_calculation": true,
    "ligand explain 2D": false,
    "PL interactions explain 2D": false,
    "PL interactions explain 3D": false,
    "optimizer": "FIRE",
    "fmax": 0.05,
    "steps": 100
  },
  "summary": {
    "total_ligands": 1,
    "successful": 1,
    "failed": 0
  },
  "results": [
    {
      "ligand_name": "ligand",
      "ligand_file": "path/to/ligand.xyz",
      "interaction_energy": -2.6674346923828125,
      "base_interaction_energy": -2.7811279296875,
      "ligand_strain_energy": 0.0687103271484375,
      "protein_strain_energy": 0.04498291015625,
      "binding_energy_kcal_mol": -61.51104400634765,
      "ligand_explainability": {
        "protein_energy_components": {
          "electrostatic_energy": -13.823354721069336,
          "zbl_repulsion": 36.19248580932617,
          "dispersion_energy": -6.992919445037842,
          "mlff_atomic_energy": -679.03564453125
        },
        "ligand_energy_components": {
          "electrostatic_energy": -1.2963955402374268,
          "zbl_repulsion": 2.4064440727233887,
          "dispersion_energy": -0.5644962787628174,
          "mlff_atomic_energy": -60.04219436645508
        },
        "complex_energy_components": {
          "electrostatic_energy": -14.730865478515625,
          "zbl_repulsion": 38.59886932373047,
          "dispersion_energy": -9.317378997802734,
          "mlff_atomic_energy": -740.4879150390625
        },
        "interaction_energy_components": {
          "electrostatic_energy": 0.3888847827911377,
          "zbl_repulsion": -6.0558319091796875e-05,
          "dispersion_energy": -1.7599632740020752,
          "mlff_atomic_energy": -1.4100761413574219
        }
      }
    }
  ]
}
```

</details>


<details>
<summary><h2>🤝 Contributing</h2></summary>

We welcome contributions! Please see our CONTRIBUTING.md (to be written) for details on:
- Code style and formatting
- Testing requirements
- Documentation standards
- Pull request process

</details>


<details>
<summary><h2>📄 License</h2></summary>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

</details>


## 📚 Citation


