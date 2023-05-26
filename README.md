# Simplex-Renormalization-Group

A toolbox for simplex path integral and renormalization group for high-order interactions

## demo

```bash
# LRG
python main.py -a LRG_BA
python main.py -a LRG_ER
python main.py -a LRG_WS
python main.py -a LRG_PROTEINS
python main.py -a LRG_ENZYMES
python main.py -a LRG_DD
python main.py -a LRG_Neuro
```

## environment

python = 3.8.3
pytorch = 1.8.0

```
conda create --name SRG python=3.8.3
conda activate SRG

# on Windows without GPU
conda install pytorch=1.8.0 torchvision=0.9.0 torchaudio=0.8.0 cpuonly -c pytorch

# on linux with GPU
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# other packages
conda install scipy
conda install scikit-learn
conda install matplotlib
conda install seaborn
conda install networkx
pip3_path_in_your_conda_env/pip3 install powerlaw # e.g. ~/miniconda3/envs/Neuro-RG/bin/pip3 install powerlaw
```
