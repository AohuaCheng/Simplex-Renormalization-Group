import numpy as np
import os.path as osp
import scipy.io as scio
import json
import networkx as nx
from configs.config_global import ROOT_DIR

class DatasetIters(object):
    def __init__(self, config):
        """
        Initialize a list of data loaders and datum sizes
        if only one dataset is specified, return a list containing one data loader
        """
        if type(config.dataset) is list or  type(config.dataset) is tuple:
            assert all([type(d_set) is str for d_set in config.dataset]
                       ), 'all dataset must be string'
            dataset_list = config.dataset
        elif type(config.dataset) is str:
            dataset_list = [config.dataset]
        else:
            raise NotImplementedError('Dataset config not recognized')
        self.num_datasets = len(dataset_list)

        self.data_sets = []
        self.data_infos = []
        for d_set in dataset_list:
            data_set, data_info = init_single_dataset(d_set, config)
            self.data_sets.append(data_set)
            self.data_infos.append(data_info)

def init_single_dataset(dataset, config):
    data_set = []
    data_info = []
    save_path = osp.join(ROOT_DIR, 'data', dataset)
    if dataset in ['neuropixel']:
        # Neuropixel data
        if config.conditions[0] == 'all':
            with open(osp.join(ROOT_DIR, 'data', dataset, 'region42.json'), 'r') as fp:
                regions = json.load(fp)
                config.conditions = regions['regions']
        for c in config.conditions:
            trials, trial_info = [], []
            for id in config.trials:
                trial = np.load(osp.join(save_path, '{}'.format(c), 'static_Corr_{}.npy'.format(id)))
                trials.append(1-trial)
                trial_info.append([c, 'static_Corr_{}'.format(id)])
            data_set.append(trials)
            data_info.append(trial_info)
    elif dataset in ['BA', 'ER']:
        # Random graph data
        for c in config.conditions:
            trials, trial_info = [], []
            for id in config.trials:
                G = generate_random_graph(dataset, config.N, c)
                trial = nx.adjacency_matrix(G).toarray()
                trials.append(trial)
                trial_info.append(['para_{}'.format(c), 'N{}_Adj_{}'.format(config.N, id)])
            data_set.append(trials)
            data_info.append(trial_info)
    else:
        raise ValueError('Dataset not found: ', dataset)

    return data_set, data_info
    
def generate_random_graph(dataset, N, para): 
    if dataset == 'ER':
        G = nx.random_graphs.erdos_renyi_graph(N, para)
    if dataset == 'BA':
        G = nx.random_graphs.barabasi_albert_graph(N, para)
        return G