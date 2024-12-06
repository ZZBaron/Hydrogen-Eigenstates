# Hydrogen Orbital Visualization

A Python-based tool for visualizing and analyzing hydrogen atomic orbitals, featuring both analytical and numerical solutions to the radial Schrödinger equation.

## Features

- Calculate and visualize hydrogen atomic orbitals for any valid quantum numbers (n, l, m)
- Compare analytical and numerical solutions of the radial Schrödinger equation
- Interactive 3D visualization with adjustable viewing angles
- Generate animations of orbital rotations
- Export orbital data to CSV files
- Support for both probability density and wavefunction visualization

## Requirements

```
numpy
scipy
matplotlib
pandas
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hydrogen-orbital-viz.git
   cd hydrogen-orbital-viz
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main functionality is provided through the `HydrogenWavefunction` class and helper functions in `H_atomic_orbitals.py`.

Basic usage example:
```python
from H_atomic_orbitals import create_hydrogen_visualization

# Create interactive visualization for n=3, l=2, m=1 orbital
create_hydrogen_visualization(n=3, l=2, m=1)

# Generate animation
create_hydrogen_visualization(n=3, l=2, m=1, save_animation=True)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.