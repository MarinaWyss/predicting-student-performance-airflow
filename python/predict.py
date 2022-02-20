import logging
import pickle

import pandas as pd

from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')


def predict_and_evaluate(configs: dict):
    """Loads the trained model and predicts on the test data. Then, 
    saves the predictions and model performance metrics.

    Args:
        configs (dict): config file
    """
    logging.info("Reading test data.")
    train_df = pd.read_csv(configs['test_data_path'])

    logging.info("Loading trained model.")
    model = pickle.load(open(onfigs['model_output_path'], 'rb'))
    
    logging.info("Predicting on test data.")
    preds = model.fit(test_df[configs['features']])

    pred_df = pd.DataFrame(
        preds,
        # Naming the columns this way in case the order changes at some point
        columns=[f'pred_{test_df[targets].columns[0]}', f'pred_{test_df[targets].columns[1]}']
    )
    test_df.loc[:, 'pred_math_score'] = pred_df['pred_math_score'].values
    test_df.loc[:, 'pred_writing_score'] = pred_df['pred_writing_score'].values

    logging.info("Saving predictions.")
    test_df.to_csv(configs['predictions_path'], index=False)

    logging.info("Calculating model performance.")
    math_mae = mean_absolute_error(
        test_df['math_score'],
        test_df['pred_math_score']
    )
    writing_mae = mean_absolute_error(
        test_df['writing_score'],
        test_df['pred_writing_score']
    )
    metrics = pd.DataFrame({
        'math_mae': math_mae,
        'writing_mae': writing_mae
    }, index=[0])

    logging.info(f" Math MAE:", math_mae)
    logging.info(f" Writing MAE:", writing_mae)

    logging.info("Saving metrics.")
    metrics.to_csv(configs['metrics_path'], index=False)

    logging.info("Pipeline complete.")
    return test_df, metrics 
