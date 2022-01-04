# ENI
Official implementation of the VR2022 paper "ENI: Quantifying Environment Compatibility for Natural Walking in Virtual Reality"

## Usage instructions

1) Install [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). `conda` is *required* due to a dependency on the `scikit-geometry` library.

2) Clone this repository.

3) Using the `conda` terminal, navigate to the top level of this repository wherever you saved it on your computer.

4) Create and activate a virtual environment using the following commands:
```bash
conda env create -f environment.yml
conda activate eni
```

5) Run ```python3 environment.py```.

The results can be found in the `img` folder.
