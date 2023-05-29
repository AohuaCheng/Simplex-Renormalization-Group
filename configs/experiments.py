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

def SRG_BA():
    config = BaseConfig()
    config.experiment_name = 'SRG_BA'
    config.analysis_type = 'SRG'
    
    config.dataset = 'BA'
    config.conditions = [2]
    config.trials = list(range(50))

    config.tau_star = .1
    config.binary = 0.6
    
    config_ranges = OrderedDict()
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs

def SRG_ER():
    config = BaseConfig()
    config.experiment_name = 'SRG_ER'
    config.analysis_type = 'SRG'
    
    config.dataset = 'ER'
    config.conditions = [0.01]
    config.trials = list(range(50))

    config.tau_star = .1
    config.binary = 0.6
    
    config_ranges = OrderedDict()
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs

def SRG_WS():
    config = BaseConfig()
    config.experiment_name = 'SRG_WS'
    config.analysis_type = 'SRG'
    
    config.dataset = 'WS'
    config.conditions = [10]
    config.trials = list(range(50))

    config.tau_star = .1
    config.binary = 0.6
    
    config_ranges = OrderedDict()
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs

def SRG_PROTEINS():
    config = BaseConfig()
    config.experiment_name = 'SRG_PROTEINS'
    config.analysis_type = 'SRG'
    
    config.dataset = 'PROTEINS'
    config.conditions = []
    config.trials = []

    config.tau_star = .2
    config.binary = 0.6
    
    config_ranges = OrderedDict()
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs

def SRG_ENZYMES():
    config = BaseConfig()
    config.experiment_name = 'SRG_ENZYMES'
    config.analysis_type = 'SRG'
    
    config.dataset = 'ENZYMES'
    config.conditions = []
    config.trials = []

    config.tau_star = .2
    config.binary = 0.6
    
    config_ranges = OrderedDict()
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs

def SRG_DD():
    config = BaseConfig()
    config.experiment_name = 'SRG_DD'
    config.analysis_type = 'SRG'
    
    config.dataset = 'DD'
    config.conditions = []
    config.trials = []

    config.tau_star = .1
    config.binary = 0.6
    
    config_ranges = OrderedDict()
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs

def SRG_Neuro():
    config = BaseConfig()
    config.experiment_name = 'SRG_Neuro'
    config.analysis_type = 'SRG'
    
    config.dataset = 'neuropixel'
    config.conditions = ['all']
    config.trials = list(range(16))

    config.tau_star = .4
    
    config_ranges = OrderedDict()
    config_ranges['binary'] = [0.8]
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs
