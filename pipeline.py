import logging
from dataprocessing import load_data, preprocess_data
from eda import run_eda
from train import train_model

logging.basicConfig(filename='pipeline.log', level=logging.INFO)

def run_pipeline():
    logging.info("Pipeline Started")

    df = load_data()
    df = preprocess_data(df)

    run_eda(df)
    train_model(df)

    logging.info("Pipeline Completed")

if __name__ == "__main__":
    run_pipeline()