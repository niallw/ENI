# ENI
Official implementation of the IEEE VR 2022 paper "[ENI: Quantifying Environment Compatibility for Natural Walking in Virtual Reality](https://gamma.umd.edu/eni/)" by [Niall L. Wiliams](https://niallw.github.io/), [Aniket Bera](https://www.cs.umd.edu/~ab/), and [Dinesh Manocha](https://www.cs.umd.edu/people/dmanocha).

## Usage instructions

1) Install [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). `conda` is *required* due to a dependency on the `scikit-geometry` library.

2) Clone this repository.

3) Using the `conda` terminal, navigate to the top level of this repository wherever you saved it on your computer.

4) Create and activate a virtual environment using the following commands:
```bash
conda env create -f environment.yml
conda activate eni
```

5) Run the script using the following command:
```bash
python3 environment.py
```

The results can be found in the `img` folder.

## Bibtex
```
@inproceedings{williams2022eni,
title={{E}{N}{I}: {Q}uantifying {E}nvironment {C}ompatibility for {N}atural {W}alking in {V}irtual {R}eality},
author={Williams, Niall L and Bera, Aniket and Manocha, Dinesh},
booktitle={2022 IEEE Virtual Reality and 3D User Interfaces (VR)},
year={2022},
organization={IEEE}
}
```