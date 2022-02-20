import logging
import pickle

import pandas as pd

from sklearn.svm import LinearSVR
from sklearn.multioutput import RegressorChain

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')


def train_and_save_model(configs: dict,
                         params: dict):
    """Trains the model using the best params from the previous hyperparam
    tuning step. Saves the trained pickled model to the model/ dir.

    Args:
        configs (dict): config file
        params (dict): dictionary of best parameters
    """
    logging.info("Reading train data.")
    train_df = pd.read_csv(configs['train_data_path'])

    logging.info("Training model")
    model = RegressorChain(LinearSVR())
    model.set_params(**params)
    model.fit(train_df[configs['features']], train_df[configs['targets']])

    logging.info("Pickling model and saving to model/ dir.")
    pickle.dump(model, open(configs['model_path'], 'wb'))

    logging.info("Model saved.")
