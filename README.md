# Predicting Student Performance - Airflow Pipeline

This repo contains code for a simple Airflow pipeline to [predict student math and writing grades](https://www.kaggle.com/spscientist/students-performance-in-exams?select=StudentsPerformance.csv) using a multivariate regression approach (sequenced support vector regression).

- **kaggle_student_performance.ipynb**: Notebook containing EDA and model experiments.
- **student_performance_dag.py**: Airflow DAG code.
- **python/**:
    - **data_prep.py**: Code to download the data from Kaggle and prepare it for model training. Saves the prepared and split train-test code to the `data/` directory.
    - **hyperparam_tune.py**: Code to run hyperparam tuning on the model. The best params are communicated to the training step via an Airflow XCOM.
    - **train_model.py**: Trains the model using the best params, pickles it, and saves it to the `model/` directory.
    - **predict.py**: Loads the trained model and predicts the targets on the test data. Saves the predictions and model metrics to the `results/` directory.
    - **configs.yaml**: Config file
    - **tests/**:
        - **test_data_prep.py**
        - **test_hyperparam_tune.py**
        - **test_train_model.py**
        - **test_predict.py**
- **data/**: Directory containing the transformed data for model training and testing.
- **model/**: Directory containing the pickled trained model.
- **results/**: Directory containing the final predictions and model metrics. 
