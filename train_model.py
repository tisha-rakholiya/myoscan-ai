# MyoScan AI - Model Training
# Team: Code Catalyst
# Tisha Rakholiya, Shreya Nath
# Bharat Academix CodeQuest 2026

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import pickle
import os

np.random.seed(42)

# ----------------------------------------------------------
# Synthetic data based on clinical muscle injury grading:
#   Grade 1 (Low Risk)      -> minor fiber damage (<10%)
#   Grade 2 (Moderate Risk) -> partial tear (10-90%)
#   Grade 3 (High Risk)     -> complete/near-complete rupture
#
# Features used:
#   pain_intensity  : 1-10
#   swelling        : 0=none, 1=mild, 2=moderate, 3=severe
#   weakness        : 0=none, 1=mild, 2=moderate, 3=severe
#   movement_limit  : 0=none, 1=mild, 2=moderate, 3=severe
#   pain_type       : 0=dull, 1=sharp, 2=burning, 3=throbbing
#   duration        : 0=<24h, 1=1-3days, 2=>3days
#   age             : 18-70
#   previous_injury : 0=no, 1=yes
#   activity_level  : 0=sedentary, 1=light, 2=moderate, 3=high
# ----------------------------------------------------------

X = []
y = []

# Grade 1 - Low Risk
for i in range(500):
    pain  = np.random.randint(1, 5)
    swell = np.random.randint(0, 2)
    weak  = np.random.randint(0, 2)
    move  = np.random.randint(0, 2)
    ptype = np.random.choice([0, 1])
    dur   = np.random.choice([0, 1])
    age  = np.random.randint(18, 60)
    prev = np.random.choice([0, 1], p=[0.8, 0.2])
    actv = np.random.choice([0, 1, 2])

    # realistic noise: sometimes low-risk patients report higher pain
    if np.random.random() < 0.15:
        pain = min(10, pain + np.random.randint(1, 3))

    X.append([pain, swell, weak, move, ptype, dur, age, prev, actv])
    y.append(0)

# Grade 2 - Moderate Risk
for i in range(500):
    pain  = np.random.randint(4, 8)
    swell = np.random.randint(1, 3)
    weak  = np.random.randint(1, 3)
    move  = np.random.randint(1, 3)
    ptype = np.random.randint(0, 4)
    dur   = np.random.randint(0, 3)
    age  = np.random.randint(20, 65)
    prev = np.random.choice([0, 1], p=[0.5, 0.5])
    actv = np.random.choice([1, 2])

    # overlap at boundaries - real medical data is never perfectly separable
    if np.random.random() < 0.12:
        pain = max(1, pain - np.random.randint(1, 3))

    X.append([pain, swell, weak, move, ptype, dur, age, prev, actv])
    y.append(1)

# Grade 3 - High Risk
for i in range(500):
    pain  = np.random.randint(7, 11)
    swell = np.random.randint(2, 4)
    weak  = np.random.randint(2, 4)
    move  = np.random.randint(2, 4)
    ptype = np.random.choice([1, 2, 3])
    dur   = np.random.randint(1, 3)
    age  = np.random.randint(30, 70)
    prev = np.random.choice([0, 1], p=[0.3, 0.7])
    actv = np.random.choice([2, 3])

    X.append([pain, swell, weak, move, ptype, dur, age, prev, actv])
    y.append(2)

X = np.array(X)
y = np.array(y)

print(f"Total samples: {len(X)}")
print(f"Low={sum(y==0)}, Moderate={sum(y==1)}, High={sum(y==2)}")

# train-test split (stratified so all classes appear in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# train model
print("\nTraining Random Forest...")
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# test accuracy
test_acc = model.score(X_test, y_test)
print(f"Test Accuracy : {test_acc * 100:.1f}%")

# 5-fold cross validation - this proves model generalizes, not just memorizes
cv = cross_val_score(model, X, y, cv=5)
print(f"Cross-Val (5-fold): {cv.mean()*100:.1f}% +/- {cv.std()*100:.1f}%")

# full classification report
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low", "Moderate", "High"]))

feature_names = [
    "Pain Intensity",
    "Swelling",
    "Weakness",
    "Movement Limitation",
    "Pain Type",
    "Duration",
    "Age",
    "Previous Injury",
    "Activity Level"
]

print("\nFeature Importance:")

for name, importance in zip(feature_names, model.feature_importances_):
    print(f"{name}: {importance*100:.2f}%")
# save model
if not os.path.exists("model"):
    os.mkdir("model")

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Saved to model/model.pkl")
print("Run: python app.py")
