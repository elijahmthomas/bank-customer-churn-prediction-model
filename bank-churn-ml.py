import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv('Bank Customer Churn Prediction.csv')
print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

print("Raw churn values:", df['churn'].value_counts(dropna=False))

df['churn'] = df['churn'].astype(str).str.strip().str.lower()
df['churn'] = df['churn'].map({'yes': 1, 'no': 0, '1': 1, '0': 0})

df = df.dropna(subset=['churn'])

print("Churn values after cleaning:", df['churn'].value_counts(dropna=False))

number_columns = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
category_columns = ['country', 'gender', 'credit_card', 'active_member']

for col in number_columns:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())
    else:
        print(f"Missing numeric column: {col}")

for col in category_columns:
    if col in df.columns:
        df[col] = df[col].fillna('unknown')
    else:
        print(f"Missing categorical column: {col}")

df = pd.get_dummies(df, columns=[col for col in category_columns if col in df.columns], drop_first=True)

print("Final shape:", df.shape)
print("Churn counts:", df['churn'].value_counts())

drop_cols = ['customer_id']
for col in drop_cols:
    if col not in df.columns:
        drop_cols.remove(col)

X = df.drop(drop_cols + ['churn'], axis=1)
y = df['churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_train, y_train)

# Get predicted probabilities (positive class)
y_proba = model.predict_proba(X_test)[:, 1]

# Default threshold predictions (0.5)
y_pred_default = (y_proba >= 0.5).astype(int)
print("Evaluation at default threshold (0.5):")
print(confusion_matrix(y_test, y_pred_default))
print(classification_report(y_test, y_pred_default))

# Custom threshold (0.3)
custom_threshold = 0.3
y_pred_custom = (y_proba >= custom_threshold).astype(int)
print(f"Evaluation at custom threshold ({custom_threshold}):")
print(confusion_matrix(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom))

# Plot precision and recall vs threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(8,5))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall vs Threshold')
plt.legend()
plt.show()

results_df = df.copy()
results_df['churn_probability'] = model.predict_proba(X_scaled)[:, 1]
results_df['predicted_churn'] = (results_df['churn_probability'] >= 0.5).astype(int)

# Save to CSV
results_df.to_csv('churn_predictions.csv', index=False)
print("Predictions exported to churn_predictions.csv")