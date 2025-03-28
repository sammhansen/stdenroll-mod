import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Simulated dataset
data = {
    "GPA": np.random.uniform(2.0, 4.0, 500),
    "Test_Score": np.random.randint(800, 1600, 500),
    "Age": np.random.randint(17, 30, 500),
    "Financial_Aid": np.random.choice([0, 1], 500),
    "Extracurriculars": np.random.choice([0, 1], 500),
    "Enrollment_Status": np.random.choice([0, 1], 500),
}

df = pd.DataFrame(data)

# Splitting data
X = df.drop(columns=["Enrollment_Status"])
y = df["Enrollment_Status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Training model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
