# Simplex-Renormalization-Group

## demo

```bash
# LRG
python main.py -a LRG_BA
python main.py -a LRG_Neuro
```

## environment

python = 3.8.3
pytorch = 1.8.0

```
conda create --name  Simplex-Renormalization-Group python=3.8.3
conda activate  Simplex-Renormalization-Group

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
```
