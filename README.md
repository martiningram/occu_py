# OccuPy: (Multi-species) Occupancy detection modelling in python

This package allows fitting multi-species occupancy detection models using
variational inference and MCMC, as well as single-species occupancy detection
models with maximum likelihood. It can be used either as a python package, or in
R through reticulate.

## Installation

Whether you are planning to use the package in R or python, the setup is
initially the same. The instructions below use the `conda` package manager in
python, which is recommended. Any package manager should work, however.

### Preliminaries: Setting up a conda virtual environment

* Step 1: Install (mini-)conda, if it is not installed already. It can be
  obtained from here: https://docs.conda.io/en/latest/miniconda.html
* Step 2: Once conda is installed, it is recommended to create a virtual
  environment. It can be created as follows:
  
```bash
conda create -n occu_py python=3
```

* Step 3: Activate the virtual environment using `conda activate occu_py`. Note
  that any name can be give, not just `occu_py`.
  
### Installing the package

Please install the following repositories:

* `ml_tools`: Easiest using: `python -m pip install git+https://github.com/martiningram/ml_tools.git`
* `jax_advi`: Easiest using: `python -m pip install git+https://github.com/martiningram/jax_advi.git`
* `numpyro`: Easiest using: `python -m pip install git+https://github.com/pyro-ppl/numpyro@8bb94f170de3f6c276fe61e4c92cd4e21de70a4b`
* The requirements listed in `requirements.txt`. You can do this using `pip install -r requirements.txt`

To get GPU support, you will need to install JAX with GPU support. Instructions
are here: https://github.com/google/jax . Please note that we have noticed a
large performance hit with the latest JAX version (0.2.12 at the time of
writing). The timings in the paper are using JAX 0.2.8 with jaxlib 0.1.57, so
please install this combination. At the time of writing, this could be done as
follows:

```
pip install --upgrade jax==0.2.8 jaxlib==0.1.57+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Once all this is installed, you can install the actual `occu_py` package: `pip install -e .`

## Running occu_py

Please see the `examples` subfolder for some example uses.
