import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from configs.config_global import FIG_DIR

if not osp.exists(FIG_DIR):
    os.makedirs(FIG_DIR)
    
def get_plot_colors(n, type='Blues'):
    if type == 'Blues':
        colors = plt.cm.Blues(np.linspace(0, 1, n))
    elif type == 'YlGnBu':
        colors = plt.cm.YlGnBu(np.linspace(0, 1, n))
    elif type == 'GnBu':
        colors = plt.cm.GnBu(np.linspace(0, 1, n))
    return colors

def get_plot_path(cfg, info):
    fig_path = osp.join(FIG_DIR, cfg.experiment_name, cfg.dataset_name, '{}_{}'.format(info[0],info[1]))
    if not osp.exists(fig_path):
        os.makedirs(fig_path)
    return fig_path