# evaluate_models.py
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------- Load Dataset -----------------
df = pd.read_csv('GJ_RJ_dataset.csv')
df.columns = df.columns.str.strip()
df.dropna(subset=['name', 'rating', 'lat', 'lng', 'reviews'], inplace=True)
df.fillna("", inplace=True)

df['combined'] = df['description']

# Fallback mood and budget columns
if 'mood' not in df.columns:
    df['mood'] = np.random.choice(['Relaxing', 'Adventurous', 'Romantic', 'Cultural', 'Spiritual'], len(df))
if 'budget' not in df.columns:
    df['budget'] = np.random.choice(['Free', 'Regular', 'Moderate', 'Premium'], len(df))

budget_map = {
    'Free': 0,
    'Regular': 300,
    'Moderate': 500,
    'Premium': 800
}
df['budget_numeric'] = df['budget'].map(budget_map)

# ----------------- Load Models -----------------
mood_model = joblib.load("models/mood_classifier.pkl")
budget_model = joblib.load("models/budget_predictor.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# ----------------- Evaluate Mood Classifier -----------------
print("\n---- Mood Classifier Evaluation ----")
X_train, X_test, y_train, y_test = train_test_split(df['combined'], df['mood'], test_size=0.2, random_state=42)
y_pred = mood_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))
print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------- Evaluate Budget Regressor -----------------
print("\n---- Budget Predictor Evaluation ----")
X_budget = tfidf.transform(df['combined'])
y_budget = df['budget_numeric']
y_pred_budget = budget_model.predict(X_budget)

print("MAE:", mean_absolute_error(y_budget, y_pred_budget))
print("MSE:", mean_squared_error(y_budget, y_pred_budget))
print("RMSE:", np.sqrt(mean_squared_error(y_budget, y_pred_budget)))
print("R² Score:", r2_score(y_budget, y_pred_budget))
