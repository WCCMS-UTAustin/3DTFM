# 3D TFM - Inverse Model for Heterogeneous Hydrogel Modulus

![demonstration](/images/hMVIC - ECM and u - Black Background.png)

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

## Usage

This code was designed to use the TFM data processed by the object-oriented version our software [FM-Track](https://github.com/elejeune11/FM-Track/tree/objectoriented).

Further processing steps on the output of FM-Track are required to create the appropriate geometry and input directory format. See TODO for an example.

For examples of how to run the inverse model, see TODO for examples. All scripts also have help messages, for instance

```
inverse --help
```

## Citing

To cite this code, please use the following BibTeX entry:

```
TODO
```

## License

TODO
