# Simplex Renormalization Group

A toolbox for simplex path integral and renormalization group for high-order interactions.

## Basic Demo to Run SRG

Open `Demo.ipynb`, and you'll see the basic procedure and analysis of simplex path integrals and simplex renormalization group. 

You can replace the example dataset (Barabasi-Albert network) with your own data (only requires the adjacency matrix defined by the relation matrix) and follow the same process in `Demo.ipynb` to obtain the Laplacians over multiple renormalization steps for arbitrary order simplices. 

## environment

python = 3.8.3
pytorch = 1.8.0

```
conda create --name SRG python=3.8.3
conda activate SRG

# main packages
conda install os
conda install copy
conda install numpy
conda install scipy
conda install scikit-learn
conda install matplotlib
conda install seaborn
conda install networkx
conda install itertools
pip3_path_in_your_conda_env/pip3 install powerlaw # e.g. ~/miniconda3/envs/Neuro-RG/bin/pip3 install powerlaw
```

## Citation
If you find our repo useful in your research, please kindly consider cite:
```
@article{cheng2023simplex,
  title={Simplex path integral and renormalization group for high-order interactions},
  author={Cheng, Aohua and Sun, Pei and Tian, Yang},
  journal={arXiv preprint arXiv:2305.01895},
  year={2023}
}
```
