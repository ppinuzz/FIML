# Datasets

- Each directory should be named after a case, e.g. `BCDirichlet_steady_nu_u`
- Each case directory should contain:
	- a `raw` directory, containing several subdirectories (one per case) with the paremeters saved in the directory name (e.g. `u00_uL2_nu00.001_cubic` has $u_0 = 0$, $u_L = 2$, $\nu_0 = 0.001$ and the `cubic` $\beta$ profile for $\nu = \nu_0 \beta$)
	- a `FIML` directory, containing a common `setup.yaml` file read by the main script `CreateFIMLdataset.py`
	- a copy of the Python file containing the problem definition, to be imported in the main script `CreateRawDataset.py`
- Each subdirectory in `raw` contains two files:
	- `solution.h5`: solution on an extremely fine mesh of the exact equation ("exact solution", not considering numerical errors)
	- `probes.h5`: exact solution sampled at several points on the domain (artificial "experimental data")