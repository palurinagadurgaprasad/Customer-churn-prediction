import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv")

# Clean data
df.replace(' ', pd.NA, inplace=True)
df.dropna(inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# Select only numeric columns for simplicity
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'churn_model.pkl')
print("✅ Model trained successfully (3 features) and saved as churn_model.pkl")
