import yaml
import logging

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')


def prepare_data(configs: dict) -> pd.DataFrame:
    """Completes the data preparation steps:
        1. reads the data from Kaggle
        2. renames columns
        3. label encodes categorical columns
        4. splits the data into training and testing sets
        5. saves the data back to the `data` dir

    Args:
        configs (dict): config file
    """
    logging.info("Reading data from Kaggle.")
    data = pd.read_csv(configs['kaggle_data_path'])

    logging.info("Renaming columns.")
    data.rename(columns={
            "race/ethnicity": "ethnicity",
            "parental level of education": "parent_education",
            "math score": "math_score",
            "reading score": "reading_score",
            "writing score": "writing_score",
            "test preparation course": "test_prep"
        }, inplace=True)

    logging.info("Label encoding features.")
    for c in configs['features']:
        data[c] = le.fit_transform(data[c])

    logging.info("Doing the train-test split.")
    train_df, test_df = train_test_split(
        data,
        test_size=configs['test_size'],
        random_state=configs['random_state']
    )

    logging.info("Saving prepared train and test data to the data/ directory.")
    train_df.to_csv(configs['train_data_path'], index=False)
    test_df.to_csv(configs['test_data_path'], index=False)

    logging.info("Data prep done.")
