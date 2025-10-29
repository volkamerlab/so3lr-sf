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
- **Structure Optimization**: Optimize protein, ligand, and complex structures with FIRE/LBFGS algorithms
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
├── structure_ops.py                   # Structure manipulation and optimization
├── explainability.py                  # Energy-based explainability analysis and visualization
├── explain_utils.py                   # Utility functions for explainability analysis
├── molecule_loader.py                 # Molecule loading and file format handling
├── utils.py                           # General utility functions and I/O operations
└── __init__.py                        # Package interface
```
### Output directory Structure
```
results_steps_100_fmax0.05/
├── results_summary.json           # Main results file
├── optimization_log.json          # Optimization details (if --opt-log)
├── individual_ligands/             # Extracted ligands (for multi-SDF)
├── opt_ligand/                     # Optimized ligand structures
├── opt_complexes/                  # Optimized complex structures
└── ligand_exp/                     # Explainability heatmaps
    ├── ligand_001_heatmap.png
    └── ligand_002_heatmap.png
```

</details>


<details>
<summary><h2>📋 Command Line Arguments</h2></summary>

### Required Arguments
- `--protein`: Path to protein structure file
- `--ligands`: Path to ligand file, directory, or multi-SDF file

### Workflow Options
- `--trim`: Trim protein around ligand(s) before calculation
- `--optimize`: Optimize structures before energy calculation
- `--exp-lig`: Generate explainability analysis and heatmaps
- `--exp-prot`: Generate protein explainability with protein-ligand interaction analysis and interacting residue coloring depending on their energy contribution
- `--exp-3d`: Generate 3D PyMOL visualization of protein energy components

### Trimming Parameters
- `--radius FLOAT`: Radius in Angstroms for protein trimming (default: 10.0)
- `--trim-lig FILE`: Specific ligand file to use for trimming

### Optimization Parameters
- `--optimizer {FIRE,LBFGS}`: Optimization algorithm (default: FIRE)
- `--fmax FLOAT`: Force convergence criterion in eV/Å (default: 0.05)
- `--steps INT`: Maximum optimization steps (default: 100)
- `--opt-radius FLOAT`: Optimization radius around ligand

### Model Parameters
- `--model-path PATH`: Path to SO3LR model parameters (auto-detected if not specified)

### Logging & Output
- `-v, --verbose`: Enable detailed logging output
- `--opt-log`: Save optimization details to JSON file

</details>


<details>
<summary><h2>🔧 Quick Start</h2></summary>

### Basic Usage

```bash
# Simple protein-ligand interaction calculation
python so3lr_sf.py --protein protein.pdb --ligands ligand.sdf

# With structure optimization
python so3lr_sf.py --protein protein.pdb --ligands ligand.sdf --optimize

# Full workflow with explainability
python so3lr_sf.py --protein protein.pdb --ligands ligands.sdf \
    --trim --optimize --exp-lig --verbose
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
    --trim \
    --radius 10.0 \
    --trim-lig ref_lig.sdf \
    --verbose
```

### Example 3: Optimized Workflow
```bash
python so3lr_sf.py \
    --protein protein.pdb \
    --ligands ligands.sdf \
    --optimize \
    --optimizer FIRE \
    --fmax 0.05 \
    --steps 100 \
    --opt-radius 4.0 \
    --verbose
```

### Example 4: Full Analysis Pipeline
```bash
python so3lr_sf.py \
    --protein protein.pdb \
    --ligands multi_ligands.sdf \
    --trim \
    --radius 8.0 \
    --optimize \
    --opt-radius 4.0 \
    --exp-lig \
    --opt-log \
    --verbose
```

### Example 5: Protein-Ligand Interaction Analysis
```bash
python so3lr_sf.py \
    --protein protein.pdb \
    --ligands ligand.sdf \
    --exp-prot \
    --verbose
```

</details>

<details>
<summary><h2>📊 Output Files</h2></summary>

### Results Summary (`results_summary.json`)
```json
{
  "workflow_parameters": {
    "protein": "proteins/protein_path.pdb",
    "ligands_source": "ligands/ligands13.xyz",
    "trim": false,
    "trim_radius": null,
    "optimize": false,
    "ligand explain 2D": false,
    "PL interactions explain 2D": false,
    "PL interactions explain 3D": true,
    "optimizer": null,
    "fmax": null,
    "steps": null
  },
  "summary": {
    "total_ligands": 1,
    "successful": 1,
    "failed": 0
  },
  "results": [
    {
      "ligand_name": "ligand_001",
      "interaction_energy": -3.50469970703125,
      "binding_energy_kcal_mol": -80.81837524414063,
      "analysis": {
        "component_totals": {
          "MLFF": -0.6717734336853027,
          "ZBL": 0.00029272645645050943,
          "Electrostatics": -0.13371722865849733,
          "Dispersion": -0.9648199365474284,
          "Total": -1.770017891190946
        },
        "ligand_explainability_heatmap": "ligand_exp/ligand_001_heatmap.png"
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


