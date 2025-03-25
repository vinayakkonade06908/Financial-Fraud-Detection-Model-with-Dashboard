import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
from utils.preprocessing import load_data, preprocess_data

# Load dataset
df = load_data("C:\\Users\\HP\\OneDrive\\Desktop\\FRAUD DETECTION\\transactions.csv")

# Check class distribution
print("🔍 Before balancing:", df["is_fraud"].value_counts(normalize=True))

# Balance dataset
df_fraud = df[df["is_fraud"] == 1]
df_safe = df[df["is_fraud"] == 0]
df_fraud_upsampled = resample(df_fraud, replace=True, n_samples=len(df_safe), random_state=42)
df = pd.concat([df_safe, df_fraud_upsampled])

# Check class distribution after balancing
print("✅ After balancing:", df["is_fraud"].value_counts(normalize=True))

# Preprocess data
df, scaler = preprocess_data(df)
print("🔍 Processed Data Sample:\n", df.head())

# Split dataset
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "C:\\Users\\HP\\OneDrive\\Desktop\\FRAUD DETECTION\\backend\\models\\fraud_model.pkl")
joblib.dump(scaler, "C:\\Users\\HP\\OneDrive\\Desktop\\FRAUD DETECTION\\backend\\models\\scaler.pkl")

print("✅ Model trained and saved successfully!")