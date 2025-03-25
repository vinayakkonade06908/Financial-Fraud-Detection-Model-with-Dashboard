import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df, scaler=None):
    """Preprocess the input data for the fraud detection model."""
    # Fill missing values with 0
    df.fillna(0, inplace=True)

    # Define required columns
    REQUIRED_COLUMNS = {'amount', 'transaction_type', 'merchant_category', 'transaction_count'}

    # Check if required columns exist
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        print(f"Error: Missing columns in input data: {missing_cols}")
        print(f"Available columns: {df.columns}")
        return None, None  # Return None to avoid processing incorrect data

    # Apply log transformation to 'amount' to handle skewness
    df['amount'] = np.log1p(df['amount'])  # log1p to handle zero values

    # Define categories for one-hot encoding
    categories = {
        'transaction_type': ['Online', 'POS', 'ATM'],
        'merchant_category': ['Retail', 'Electronics', 'Food', 'Clothing', 'Groceries', 'Luxury', 'Travel', 'Restaurants', 'Pharmacy']
    }

    # One-hot encode categorical columns
    for col, cats in categories.items():
        df = pd.get_dummies(df, columns=[col], prefix=col, prefix_sep='_')
        # Ensure all category columns exist (set missing ones to 0)
        for cat in cats:
            col_name = f"{col}_{cat}"
            if col_name not in df.columns:
                df[col_name] = 0  # Add missing columns with 0s

    # Normalize 'amount' and 'transaction_count'
    if scaler is None:
        print("Creating a new StandardScaler...")
        scaler = StandardScaler()
        df[['amount', 'transaction_count']] = scaler.fit_transform(df[['amount', 'transaction_count']])
    else:
        df[['amount', 'transaction_count']] = scaler.transform(df[['amount', 'transaction_count']])

    # Final checks and return
    print("Processed dataframe shape:", df.shape)
    return df, scaler