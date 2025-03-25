# Financial-Fraud-Detection-Model-with-Dashboard
Financial Fraud Detection Model using Dashboard: This project is a machine learning-based fraud detection system integrated with a Streamlit dashboard. Users input transaction details such as amount, transaction type, merchant category, and transaction count. The backend, built with FastAPI, processes this data and predicts whether the transaction is fraudulent or safe using a trained machine learning model.

# Features
✅ User-friendly Dashboard – Built with Streamlit for easy interaction.
✅ Machine Learning-powered Fraud Detection – Uses trained models to classify transactions.
✅ FastAPI Backend – Ensures efficient processing of user input.
✅ Real-time Predictions – Get instant fraud detection results.

# Technologies Used :
Python – Programming language.
Streamlit – Frontend dashboard framework.
FastAPI – Backend for API communication.
Scikit-learn / TensorFlow – Machine learning model development.
Pandas & NumPy – Data preprocessing and analysis.
Uvicorn – ASGI server for FastAPI.

# 📦 Installation :
1. Clone the repo:
   git clone https://github.com/your-username/fraud-detection-system.git
   cd fraud-detection-system
   
3. Install dependencies:
   pip install -r requirements.txt
   
# 🛠 Usage
1. Start the FastAPI backend:
   uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
   
2. Run the Streamlit dashboard frontend:
   streamlit run app.py
   
3. Access:
   - API: `http://localhost:8000/docs`
   - Dashboard: `http://localhost:8501`

# Model Details
- Algorithm: Random Forest Classifier
- Key Metrics:
  - Precision: 0.92
  - Recall: 0.89
  - F1-score: 0.90
- Features: Amount, Transaction Type, Merchant Category, Transaction Count







