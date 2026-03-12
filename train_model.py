import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# ── 1. Load Dataset ──────────────────────────────────────────────────────────
print("=" * 55)
print("       DIABETES PREDICTION MODEL — TRAINING")
print("=" * 55)

df = pd.read_csv("diabetes.csv")

# ── 2. Basic Dataset Information ─────────────────────────────────────────────
print("\n📋 Dataset Shape:", df.shape)
print("\n📊 First 5 Rows:")
print(df.head().to_string())
print("\n📈 Statistical Summary:")
print(df.describe().to_string())
print("\n🔍 Missing Values:")
print(df.isnull().sum().to_string())
print("\n🎯 Target Distribution (Outcome):")
print(df["Outcome"].value_counts().to_string())

# ── 3. Separate Features and Target ──────────────────────────────────────────
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

print(f"\n✅ Features (X): {list(X.columns)}")
print(f"✅ Target  (y): Outcome  ({y.nunique()} classes)")

# ── 4. Train / Test Split  (80 / 20) ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n📂 Training samples : {X_train.shape[0]}")
print(f"📂 Testing  samples : {X_test.shape[0]}")

# ── 5. Train Logistic Regression ──────────────────────────────────────────────
print("\n⚙️  Training Logistic Regression model …")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("✅ Training complete!")

# ── 6. Evaluate Model ────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy on Test Set: {acc * 100:.2f}%")

# ── 7. Save Model with Pickle ─────────────────────────────────────────────────
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n💾 Model saved as  diabetes_model.pkl")
print("\n" + "=" * 55)
print("  Run  →  streamlit run app.py  to launch the app!")
print("=" * 55 + "\n")