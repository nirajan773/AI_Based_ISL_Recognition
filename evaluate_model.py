
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).parent

# Load trained model
model = joblib.load(
    BASE_DIR / "models" / "isl_random_forest.joblib"
)

# Load separate testing dataset
test_data = pd.read_csv(
    BASE_DIR / "processed" / "test.csv"
)

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

print("Testing samples:", len(X_test))
print("Features:", X_test.shape[1])

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n================================")
print("       TEST RESULTS")
print("================================")

print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions,
        labels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        zero_division=0
    )
)

# Confusion matrix
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    )
)
