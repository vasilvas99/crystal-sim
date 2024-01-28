# Crystal Sim

TODO: Improve readme

## Installation

1. Install [FeniCSx](https://github.com/FEniCS/dolfinx?tab=readme-ov-file#installation)
2. pip install -r requirements.txt

## Running

```shell
mpirun -np <NUMBER_OF_PROCESSES> python3 simulator.py
```

## Known issues

FeniCSx is not available in venvs since it's installed as a global package...