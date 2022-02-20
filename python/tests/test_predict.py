import unittest
import unittest.mock as mock

import yaml

from predict import predict_and_evaluate


class TestPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open('../python/configs.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        cls.configs = configs
        cls.test_columns = ['gender', 'ethnicity', 'parent_education', 'lunch', 'test_prep',
            'math_score', 'reading_score', 'writing_score', 'pred_math_score', 'pred_writing_score']
        cls.metrics_columns = ['math_mae', 'writing_mae']

    def test_predict_and_evaluate(self):
        """
        TODO should be using a static test dataset vs. the real Kaggle data here.
        """
        metrics, test_df = predict_and_evaluate(self.configs)

        self.assertEqual(metrics.shape, (1, 2))
        self.assertEqual(test_df.shape, (300, 10))

        # no missing values
        self.assertEqual(test_df.dropna().shape, (300, 10))
    
        self.assertEqual(set(metrics.columns), set(self.metrics_columns))
        self.assertEqual(set(test_df.columns), set(self.test_columns))

        assert test_df['pred_math_score'].between(0, 100).all()
        assert test_df['pred_writing_score'].between(0, 100).all()


if __name__ == '__main__':
    unittest.main()