"""Experiments and corresponding analysis.
format adapted from https://github.com/gyyang/olfaction_evolution

Each experiment is described by a function that returns a list of configurations
function name is the experiment name

combinatorial mode:
    config_ranges should not have repetitive values
sequential mode:
    config_ranges values should have equal length,
    otherwise this will only loop through the shortest one
"""
from collections import OrderedDict
from configs.configs import BaseConfig
from utils.config_utils import vary_config

def LRG_BA():
    config = BaseConfig()
    config.experiment_name = 'LRG_BA'
    config.analysis_type = 'LRG'
    
    config.dataset = 'BA'
    config.conditions = [2]
    config.trials = list(range(50))

    config.tau_star = 1.26
    
    config_ranges = OrderedDict()
    config_ranges['binary'] = [0.6]
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs

def LRG_Neuro():
    config = BaseConfig()
    config.experiment_name = 'LRG_Neuro'
    config.analysis_type = 'LRG'
    
    config.dataset = 'neuropixel'
    config.conditions = ['all']
    config.trials = list(range(16))

    config.tau_star = .4
    
    config_ranges = OrderedDict()
    config_ranges['binary'] = [0.8]
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs
