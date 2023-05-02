import argparse

from configs.configs import BaseConfig
import configs.experiments as experiments
from utils.config_utils import save_config, configs_dict_unpack
from analyze import dataset_analyze

def analyze_experiment(experiment):
    print('Analyzing {:s} experiment'.format(experiment))
    if (experiment) in dir(experiments):
        # Get list of configurations from experiment function
        exp_configs = getattr(experiments, experiment)()
    else:
        raise ValueError('Experiment config not found: ', experiment)
    
    exp_configs = configs_dict_unpack(exp_configs)
    assert isinstance(exp_configs[0], BaseConfig), \
            'exp_configs should be list of configs'
            
    for config in exp_configs:
        save_config(config, config.save_path)
        dataset_analyze(config)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--analyze', nargs='+', help='Analyze datasets', default=[])
    args = parser.parse_args()
    experiment2analyze = args.analyze
    
    for exp in experiment2analyze:
        analyze_experiment(exp)