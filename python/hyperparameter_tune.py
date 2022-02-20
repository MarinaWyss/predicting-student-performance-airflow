import logging

import pandas as pd

from sklearn.svm import LinearSVR
from sklearn.multioutput import RegressorChain
from sklearn.model_selection import GridSearchCV

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')


def tune_hyperparams(configs: dict) -> dict:
    """Tunes the hyperparameters for a sequenced support vector regression.

    Args:
        configs (dict): config file

    Returns:
        (dict): params from the best tuning run. Will be passed
            to the next task via an XCOM.
    """
    logging.info("Reading train data.")
    train_df = pd.read_csv(configs['train_data_path'])
    
    logging.info("Running Grid Search.")
    svm_cv = GridSearchCV(
        # Selected GridSearchCV instead of random search since the data 
        # and param space are small, so why not
        estimator=RegressorChain(LinearSVR()),
        # TODO set up a sleek way to put these into the configs
        param_grid={
                'base_estimator__C': [float(x) for x in np.linspace(start=0, stop=1, num=20)],
                'base_estimator__dual': [True, False],
                'base_estimator__epsilon':  [float(x) for x in np.linspace(start=0, stop=1, num=20)],
                'base_estimator__fit_intercept': [True, False],
                'base_estimator__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            }
    )

    svm_cv.fit(train_df[configs['features']], train_df[configs['targets']])

    logging.info("Tuning done. Returning best params.")
    return svm_cv.best_params_
