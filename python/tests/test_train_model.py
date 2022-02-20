import unittest
import unittest.mock as mock

import yaml

from train_model import train_and_save_model


class TestTrainModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open('../python/configs.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        cls.configs = configs

    @mock.patch('sklearn.model.fit')
    @mock.patch('pickle.dump')
    def test_train_and_save_model(self, fit_model, pickle_dump):
        train_and_save_model(self.configs, params=dict())
        fit_model.assert_called_once()
        pickle_dump.assert_called_once()


if __name__ == '__main__':
    unittest.main()