import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle
import os

# 1. Load Data
df = pd.read_csv('customer_data.csv')

# 2. Preprocessing
le = LabelEncoder()
df['contract_type'] = le.fit_transform(df['contract_type'])

X = df[['tenure_months', 'monthly_charges', 'total_charges', 'support_tickets', 'contract_type']]
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train XGBoost Model
model = xgb.XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Save Model & Encoders in 'models' folder
os.makedirs('models', exist_ok=True)
with open('models/churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Model successfully trained and saved inside models folder!")