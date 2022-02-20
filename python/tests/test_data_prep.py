import unittest
import unittest.mock as mock

import yaml
import pandas as pd

from data_prep import prepare_data


class TestDataPrep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open('../python/configs.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        cls.configs = configs
        cls.columns = ['gender', 'ethnicity', 'parent_education', 'lunch', 'test_prep',
            'math_score', 'reading_score', 'writing_score']

    @mock.patch('pd.to_csv')
    def test_final_data(self, to_csv):
        """This test calls the data from Kaggle, which isn't ideal since that could change.
        
        But, it's a placeholder for now.
        """
        train_df, test_df = prepare_data(self.configs)

        self.assertEqual(train_df.shape, (700, 8))
        self.assertEqual(test_df.shape, (300, 8))
        self.assertEqual(set(train_df.columns), set(self.columns))
        self.assertEqual(set(test_df.columns), set(self.columns))
    
        # assert all data is numeric
        assert train_df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
        assert test_df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
        
        # assert data saved properly
        self.assertEqual(to_csv.call_count, 2)


if __name__ == '__main__':
    unittest.main()