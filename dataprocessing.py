from train import train_model
import pandas as pd
import logging
from eda import run_eda

# Logging setup (IMPORTANT for assignment)
logging.basicConfig(filename='pipeline.log', level=logging.INFO)

def load_data():
    logging.info("Loading data...")
    df = pd.read_csv("data/train_u6lujuX_CVtuZ9i.csv")
    return df

def preprocess_data(df):
    logging.info("Preprocessing started...")

    # Drop ID column
    df.drop('Loan_ID', axis=1, inplace=True)

    # Handle missing values
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

    # Fix Dependents
    # Handle Dependents (even if you didn’t before)
    df['Dependents'] = df['Dependents'].replace('3+', 3)
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Dependents'] = pd.to_numeric(df['Dependents'])

    # Encode categorical
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

    for col in cols:
        df[col] = le.fit_transform(df[col])

    # Encode target
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    logging.info("Preprocessing completed")
    return df

if __name__ == "__main__":
    df = load_data()
    # print(df.head())

    # print("Missing Values Check")
    # print(df.isnull().sum())
    df = preprocess_data(df)

    run_eda(df)
    # print(df.head())
    # print(df.isnull().sum())
    metrics = train_model(df)

    print(metrics)
    