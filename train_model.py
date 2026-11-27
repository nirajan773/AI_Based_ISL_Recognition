
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "processed"

TRAIN_FILE = PROCESSED_DIR / "train.csv"
VALIDATION_FILE = PROCESSED_DIR / "validation.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading dataset...")

train_data = pd.read_csv(TRAIN_FILE)
validation_data = pd.read_csv(VALIDATION_FILE)

# Last column contains alphabet labels.
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

X_val = validation_data.iloc[:, :-1]
y_val = validation_data.iloc[:, -1]


# ==========================================
# CHECK DATA
# ==========================================

print("\nDataset information")
print("-------------------")

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Features:", X_train.shape[1])

print("Training letters:", sorted(y_train.unique()))

if X_train.shape[1] != 126:
    raise ValueError("Expected exactly 126 landmark features.")

if set(y_train.unique()) != set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    raise ValueError("Training dataset does not contain all A-Z labels.")

if set(y_val.unique()) != set(y_train.unique()):
    raise ValueError("Training and validation classes do not match.")


# ==========================================
# CREATE RANDOM FOREST MODEL
# ==========================================

print("\nCreating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)


# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining started...")
print("Please wait...\n")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# VALIDATION
# ==========================================

print("\nEvaluating validation data...")

predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)

print("\n================================")
print("       VALIDATION RESULTS")
print("================================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_val,
        predictions,
        labels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        zero_division=0
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

model_path = MODEL_DIR / "isl_random_forest.joblib"

joblib.dump(model, model_path)

print("\n================================")
print("MODEL SAVED SUCCESSFULLY!")
print("================================")

print("Location:", model_path)
print("================================")
