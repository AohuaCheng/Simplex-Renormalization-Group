import numpy as np

from configs.config_global import NP_SEED
from datasets.data_sets import DatasetIters
from analysis.LRG import multi_LRG

def dataset_analyze(config):
    np.random.seed(NP_SEED + config.seed)
    
    # initialize dataset
    Data = DatasetIters(config)
    
    # analysis
    for data_set, data_info in zip(Data.data_sets, Data.data_infos):
        for trials, trial_info in zip(data_set, data_info):
            for trial, info in zip(trials, trial_info):
                # Preprocess
                if config.analysis_type in ['LRG']:
                    exp_data = trial
                
                if config.binary:
                    exp_data = (exp_data>config.binary).astype(np.float_)
                
                # analysis
                if 'LRG' in config.analysis_type:
                    for i in range(exp_data.shape[0]):
                        exp_data[i,i] = 0
                    mLRG = multi_LRG(config, info, exp_data)
                    for d in range(1, 4):
                        mLRG.LRG(d)
                        mLRG.LRG_plot(config, info, d)
                        mLRG.reset()

            if 'LRG' in config.analysis_type:
                for d in range(1, 4):
                    mLRG.LRG_condition_plot(config, trial_info, d)

        if 'LRG' in config.analysis_type:
            for d in range(1, 4):
                mLRG.LRG_allplot(config, data_info, d)
