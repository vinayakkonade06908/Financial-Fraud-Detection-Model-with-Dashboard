import pandas as pd
from preprocessing import preprocess_data

# Load the dataset
df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\FRAUD DETECTION\\transactions.csv")

# Preprocess the data
df, scaler = preprocess_data(df)

# Check if preprocessing was successful
if df is None:
    print("❌ Preprocessing failed! Check the input data for missing columns.")
else:
    print("✅ Preprocessing successful!")
    print(df.head())