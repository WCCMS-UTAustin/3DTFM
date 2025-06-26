# 3D TFM - Inverse Model for Heterogeneous Hydrogel Modulus

![demonstration](/images/hMVIC%20-%20ECM%20and%20u%20-%20Black%20Background.png)

A Python FEniCS implementation of our sophisticated approach for inverse modeling of highly spatially-varying modulus in 3D Traction Force Microscopy.

## Dependencies

1. Linux
2. A distribution of conda, for instance [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux)

### Optional

1. For computing normal vectors of meshes with Tensorflow: [cuda](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
2. For either processing z-stack images from scratch or preparing inputs to this software: The object-oriented version of [FM-Track](https://github.com/elejeune11/FM-Track/tree/objectoriented)

## Installation

While in the repository directory, if have cuda:

```
conda env create -f conda_env.yml
```

Otherwise, (don't have cuda):

```
conda env create -f conda_env_no_cuda.yml
```

In any case, afterwards:

```
conda activate newestfenics
conda install -c bioconda tbb=2020.2
pip install git+https://github.com/g-peery/sanity_research_utils.git git+https://github.com/g-peery/moola.git
pip install -e .
```

### Optional

To build documentation locally, install pdoc:

```
pip install pdoc
```

## Usage

This code was designed to use the TFM data processed by the object-oriented version our software [FM-Track](https://github.com/elejeune11/FM-Track/tree/objectoriented).

Further processing steps on the output of FM-Track are required to create the appropriate geometry and input directory format, see the documentation and Jupyter notebook examples.

For examples of how to run the inverse model, see `reproduction/fenics_environment_analysis.ipynb`. All scripts also have help messages, for instance

```
inverse --help
```

### Documentation

Official documentation is available, generated with [pdoc](https://pdoc.dev/).

To host a local version of documentation:

```
pdoc gel --math
```

## Citing

If you find this method useful for your research, please use the following BibTeX entry:

```
TODO
```

### Previous Approach

This code was built up starting from our previous approach, see the Supplementary Materials from the below citation:

```
@article{KHANG2023123,
    title = {Estimation of aortic valve interstitial cell-induced 3D remodeling of poly(ethylene glycol) hydrogel environments using an inverse finite element approach},
    journal = {Acta Biomaterialia},
    volume = {160},
    pages = {123-133},
    year = {2023},
    issn = {1742-7061},
    doi = {https://doi.org/10.1016/j.actbio.2023.01.043},
    url = {https://www.sciencedirect.com/science/article/pii/S174270612300051X},
    author = {Alex Khang and John Steinman and Robin Tuscher and Xinzeng Feng and Michael S. Sacks},
    keywords = {Aortic valve interstitial cell, Hydrogel, Degradation, Stiffening, 3D traction force microscopy, Computational modeling, Inverse modeling, Adjoint method, Collagen deposition},
}
```

## License

TODO

## Data

Human MVICs were isolated from consented patients undergoing mitral valve replacement by the Columbia Biobank for Translational Science (IRB-AAAR6796). The deidentified MVICs were then shipped to the University of Texas where culturing and imaging occurred (IBC-2023-00293).

This data was used for creating geometry and boundary conditions for the test problem. Relevant test problem data available here: TODO

