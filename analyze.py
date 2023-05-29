import numpy as np

from configs.config_global import NP_SEED
from datasets.data_sets import DatasetIters
from analysis.SRG import SRG

def dataset_analyze(config):
    np.random.seed(NP_SEED + config.seed)
    
    # initialize dataset
    Data = DatasetIters(config)
    
    # analysis
    for data_set, data_info in zip(Data.data_sets, Data.data_infos):
        for trials, trial_info in zip(data_set, data_info):
            for trial, info in zip(trials, trial_info):
                # Preprocess
                if config.analysis_type in ['SRG']:
                    exp_data = trial
                
                if config.binary:
                    exp_data = (exp_data>config.binary).astype(np.float_)
                
                # analysis
                if 'SRG' in config.analysis_type:
                    for i in range(exp_data.shape[0]):
                        exp_data[i,i] = 0
                    mSRG = SRG(config, info, exp_data)
                    for n in range(1, 4):
                        mSRG.SRG(n)
                        mSRG.SRG_plot(config, info, n)
                        mSRG.reset()

            if 'SRG' in config.analysis_type:
                for n in range(1, 4):
                    mSRG.SRG_condition_plot(config, trial_info, n)

        if 'SRG' in config.analysis_type:
            for n in range(1, 4):
                mSRG.SRG_allplot(config, data_info, n)
