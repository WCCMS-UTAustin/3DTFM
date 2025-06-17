# 3D TFM - Inverse Model for Heterogeneous Hydrogel Modulus

## Dependencies

conda (ie. miniconda0)

## Optional Dependencies

For computing normal vectors of meshes with Tensorflow: cuda

## Installation

While in repository directory, if have cuda:

```
conda env create -f conda_env.yml
```

Otherwise, if don't have cuda:

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

