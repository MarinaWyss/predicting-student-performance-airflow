import unittest
import unittest.mock as mock

import yaml

from hyperparameter_tune import tune_hyperparams


class TestHyperparamTune(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open('../python/configs.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        cls.configs = configs

    @mock.patch('sklearn.model_selection.GridSearchCV')
    @mock.patch('sklearn.model.fit')
    def test_tune_hyperparams(self, grid, fit_model):
        tune_hyperparams(self.configs)
        grid.assert_called_once()
        fit_model.assert_called_once()


if __name__ == '__main__':
    unittest.main()