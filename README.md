# **SO3LR-SF** - Advancing computational drug discovery with machine learning force fields.
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/hamzaibrahim21/so3lr-sf/branch/main/graph/badge.svg)](https://codecov.io/gh/hamzaibrahim21/so3lr-sf)
[![CI](https://github.com/hamzaibrahim21/so3lr-sf/workflows/CI/badge.svg)](https://github.com/hamzaibrahim21/so3lr-sf/actions)

SO3LR-SF is a comprehensive Python package for calculating protein-ligand interaction energies using SO3LR machine learning force fields. It provides advanced features including structure optimization, explainability analysis, and molecular visualization.

## 🚀 Features

### Core Functionality
- **Protein-Ligand Interaction Energy Calculation**: Calculate binding energies using SO3LR force fields
- **Structure Optimization**: Optimize protein, ligand, and complex structures with FIRE/LBFGS algorithms
- **Protein Trimming**: Trim protein structures around ligands to reduce computational cost
- **Explainability Analysis**: Generate per-atom energy contributions and molecular heatmaps
- **Multi-Ligand Screening**: Process multiple ligands from SDF files or directories

### File Format Support
- **Input**: PDB, XYZ, SDF, MOL, MOL2 formats
- **Multi-molecule files**: Automatic splitting of multi-ligand SDF or XYZ files
- **Output**: JSON results, XYZ optimized structures, PNG heatmaps

### Advanced Features
- **Silent/Verbose Modes**: Configurable logging levels for production and debugging
- **Progress Tracking**: Real-time progress bars for multi-ligand processing
- **Error Handling**: Robust error handling with detailed reporting
- **Optimization Logging**: Detailed optimization trajectories and convergence data

## 📦 Installation

@TODO

## 🔧 Quick Start

### Basic Usage

```bash
# Simple protein-ligand interaction calculation
python run_so3lr_sf.py --protein protein.pdb --ligands ligand.sdf

# With structure optimization
python run_so3lr_sf.py --protein protein.pdb --ligands ligand.sdf --optimize

# Full workflow with explainability
python run_so3lr_sf.py --protein protein.pdb --ligands ligands.sdf \
    --trim --optimize --explain --verbose
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

## 📋 Command Line Arguments

### Required Arguments
- `--protein`: Path to protein structure file
- `--ligands`: Path to ligand file, directory, or multi-SDF file

### Workflow Options
- `--trim`: Trim protein around ligand(s) before calculation
- `--optimize`: Optimize structures before energy calculation
- `--explain`: Generate explainability analysis and heatmaps

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

## 💡 Examples

### Example 1: Basic Calculation
```bash
python run_so3lr_sf.py \
    --protein tests/test_data/alanine.xyz \
    --ligands tests/test_data/water.sdf
```

### Example 2: Optimized Workflow
```bash
python run_so3lr_sf.py \
    --protein protein.pdb \
    --ligands ligands.sdf \
    --optimize \
    --optimizer FIRE \
    --fmax 0.05 \
    --steps 100 \
    --verbose
```

### Example 3: Full Analysis Pipeline
```bash
python run_so3lr_sf.py \
    --protein protein.pdb \
    --ligands multi_ligands.sdf \
    --trim \
    --radius 8.0 \
    --optimize \
    --explain \
    --opt-log \
    --verbose
```

### Example 4: Multi-Ligand Screening
```bash
python run_so3lr_sf.py \
    --protein target.pdb \
    --ligands ligand_library/ \
    --trim \
    --optimize \
    --explain \
    --verbose
```

## 📊 Output Files

### Results Summary (`results_summary.json`)
```json
{
  "workflow_parameters": {
    "protein": "protein.pdb",
    "ligands_source": "ligands.sdf",
    "trim": true,
    "optimize": true,
    "explain": true
  },
  "summary": {
    "total_ligands": 10,
    "successful": 9,
    "failed": 1
  },
  "results": [
    {
      "ligand_name": "ligand_001",
      "interaction_energy": -0.308,
      "binding_energy_kcal_mol": -7.11,
      "analysis": {
        "component_totals": {
          "MLFF": -0.489,
          "Electrostatics": 0.283,
          "Dispersion": -0.014
        },
        "heatmap_path": "ligand_exp/ligand_001_heatmap.png"
      }
    }
  ]
}
```

### Directory Structure
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


### Code Quality Metrics
- **Test Coverage**: 85%+ across all modules
- **Code Style**: Black, isort, flake8 compliant
- **Type Checking**: mypy validated
- **Documentation**: 100% function documentation coverage

### Performance Benchmarks
@TODO

## 🏗️ Architecture

### Module Organization
```
src/
├── calculator.py           # SO3LR calculator implementation
├── interaction_energy.py   # Main energy calculation functions
├── structure_ops.py        # Structure manipulation and optimization
├── explainability.py       # Explainability analysis and visualization
├── utils.py                # Utility functions and I/O operations
└── __init__.py            # Package interface
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code style and formatting
- Testing requirements
- Documentation standards
- Pull request process


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

## 🆘 Support

- **Documentation**: [Full API Documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/user/so3lr-sf/issues)
- **Discussions**: [GitHub Discussions](https://github.com/user/so3lr-sf/discussions)



