## sella-tutorial

Minimal examples for transition state (TS) search and reaction pathway calculations (NEB/IRC) using Sella, ASE, and FAIRChem. The examples cover the rotational barrier of ethane (C2H6) and a small RDKit-generated molecule (referred to as MDA).

### Contents
- `ethane-neb.py` / `ethane-neb.ipynb`: NEB + Sella TS search for ethane rotation
- `ethane-irc.py` / `ethane-irc.ipynb`: IRC tracing from the ethane TS and visualization
- `mda-neb.py` / `mda-neb.ipynb`: NEB + Sella TS search for a small RDKit-generated molecule
- `mda-irc.py` / `mda-irc.ipynb`: IRC tracing from that molecule's TS and visualization

Outputs are saved in:
- `traj/...`: structures (ASE Trajectory)
- `log/...`: logs (optimization, NEB, IRC progress)
- `out/*.gif`: animations of the reaction pathway (e.g., along the IRC)

Notebooks are paired with `.py` via `jupytext` (see `jupytext.toml`).

## Environment and installation

### Requirements
- Python 3.10+ (3.12 recommended; kernel name in examples is `fairchem312`)
- CUDA-capable GPU is recommended for speed (CPU works as well)

### Install dependencies
After creating a virtual environment, install the main dependencies:

```bash
pip install -r requirements.txt
```

Notes:
- RDKit may require `rdkit-pypi` on pip, or installation via Conda (`conda install -c rdkit rdkit`).
- Colored terminal logs are optional (install `colorlog` to enable).

### FAIRChem/UMA inference model
Each script loads a FAIRChem UMA inference unit:

```python
predictor = load_predict_unit(Path(os.environ["HOME"]) / "uma/checkpoints/uma-s-1p1.pt", device="cuda")
```

- Place the checkpoint `uma-s-1p1.pt` under `$HOME/uma/checkpoints/`.
- If you do not have a GPU, change `device="cpu"`.

## Usage

### Run as scripts
Example: NEB followed by TS search for ethane.

```bash
python ethane-neb.py
```

Example: IRC from the ethane TS (both forward and reverse directions).

```bash
python ethane-irc.py
```

Examples: NEB / IRC for the RDKit-generated molecule.

```bash
python mda-neb.py
python mda-irc.py
```

After execution, trajectories go to `traj/`, logs to `log/`, and GIFs to `out/`. GUI visualization via `ase.visualize.view` may not open a window depending on your environment; in that case, refer to the generated GIFs.

### Run with notebooks
Open the `.ipynb` files directly, or keep them synced with `.py` using `jupytext`:

```bash
pip install jupytext
jupytext --sync ethane-neb.py
```

## Troubleshooting
- RDKit installation fails: try `pip install rdkit-pypi` or Conda `conda install -c rdkit rdkit`.
- No CUDA/GPU: set `device="cpu"` (will be slower).
- Model not found: verify the path `~/uma/checkpoints/uma-s-1p1.pt` or change the path in the code accordingly.

## License
This repository is released under the MIT License. See `LICENSE`.
