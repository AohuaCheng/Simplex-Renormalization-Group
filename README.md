# Simplex Renormalization Group

A toolbox for simplex path integral and renormalization group for high-order interactions.

## Structure of Our Code

To ensure universality and scalability, our code is organized in a modular way. The `main.py` file functions as the primary entry point for our program, reading the experiment settings and performing analysis for different dataset groups. The specific grouping method can be found in `utils/config_utils.py`.

The experiment settings are stored in the `/configs` folder. The file 'config_global.py' in this folder contains global basic settings, such as file save locations and random seeds, while `configs.py` includes basic experimental parameter settings like datasets, experimental conditions, and renormalization parameters. `experiments.py` contains the actual experimental parameters, which inherit from `configs.py`. If you need to use this code for new experiments, define new experimental conditions here.

Additionally, the `/data` folder stores the raw data for the experiment, and `/datasets/data_sets.py` defines the way to call the raw data and convert the actual input data into graph adjacency matrices.

`main.py` will call `analyze.py` to perform a specific analysis on a set of experimental data we have set. Here, you can adjust the order of the simplex and the graphics to be drawn. The core code of SRG can be found in `analysis/SRG.py`, where we placed all functions involved in the entire SRG in the SRG class. The most crucial function is the SRG function, which is entirely consistent with the process described in the "Simplex renormalization group" section of our article.

Finally, all experimental data and figures will be saved in the `/experiments` and `/figures` directories.

## Basic Demo to Run SRG

Open `demo.ipynb`, and you'll see the basic procedure from preparing the dataset to plotting SRG results. 

You can replace the example dataset (BA graph) with your own data (provide the adjacency matrix at least) and follow the same process in this Jupyter Notebook to get the graphs and Laplacians over multiple renormalization steps for different order simplices. 

Note that finding all K-simplices of an n-vertices graph has a time complexity of O(2^(n/2)) and a space complexity of O(n^2) to store the graph. As such, we've applied the most efficient method (depth-first searching) that we know of to find all simplices, but running SRG for larger vertices and denser connections in a graph may still take a longer time.

## Pipeline to Run Our Experiments

- First, ensure you've set up the environment as described below. 

- Then run the following commands in your terminal to execute the experiments:
```bash
# SRG
python main.py -a SRG_BA
python main.py -a SRG_ER
python main.py -a SRG_WS
python main.py -a SRG_PROTEINS
python main.py -a SRG_ENZYMES
python main.py -a SRG_DD
python main.py -a SRG_Neuro
```
- Finally, check all results stored in the `/experiments` and `/figures` directories.

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
