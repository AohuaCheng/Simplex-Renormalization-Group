"""
Configurations for the project
format adapted from https://github.com/gyyang/olfaction_evolution

Be aware that the each field in the configuration must be in basic data type that
jason save and load can preserve. each field cannot be complicated data type
"""

class BaseConfig(object):
    def __init__(self):
        """
        dataset: dataset, eg. "neuropixel"
        """
        self.experiment_name = None
        self.analysis_type = None
        self.dataset = None
        self.save_path = None
        
        self.conditions = []
        self.trials = []
        
        self.binary = False
        self.split = False
        
        # LRG parameters
        self.N = 1000
        self.tau_star = 1.26

    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

    def __str__(self):
        return str(self.__dict__)
