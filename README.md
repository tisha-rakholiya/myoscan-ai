# MyoScan AI
## AI-Assisted Muscle Injury Risk Screening Tool

**Team Code Catalyst | Bharat Academix CodeQuest 2026**  
**Tisha Rakholiya, Shreya Nath**  
**ITM (SLS) Baroda University | BTech CSE (AI/DS) — 3rd Year**

---

## Problem Statement

Muscle damage diagnosis in India costs **₹5,000–₹20,000 per test** (EMG/MRI) and is largely inaccessible in rural and Tier-2/3 areas, affecting millions who need early detection.

---

## Solution

MyoScan AI is a **browser-based AI risk-screening tool** that accepts user-reported symptoms and classifies muscle injury risk into **Low / Moderate / High** categories using a trained Machine Learning model — at zero cost, in under 2 minutes.

> **Scope:** This is a triage/screening aid — not a replacement for professional medical diagnosis.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | Python, Scikit-learn (Random Forest Classifier) |
| Data Processing | NumPy |
| Backend | Flask (Python Web Framework) |
| Frontend | HTML5, CSS3, JavaScript |
| Deployment | Localhost / Render / Heroku |

---

## How to Run Locally

### Prerequisites
- Python 3.8 or above
- pip

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the ML model
```bash
python train_model.py
```

### Step 3 — Start the web app
```bash
python app.py
```

### Step 4 — Open in browser
```
http://127.0.0.1:5000
```

---

## Project Structure

```
myoscan-ai/
├── app.py              # Flask web server + prediction logic
├── train_model.py      # ML model training script
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── model/
│   └── model.pkl       # Trained classifier (auto-generated)
└── templates/
    ├── index.html      # Symptom assessment form
    └── result.html     # Risk result & recommendations page
```

---

## How the ML Model Works

**Input Features (6):**
1. `pain_intensity` — Self-rated pain on scale 1–10
2. `swelling` — 0=None, 1=Mild, 2=Moderate, 3=Severe
3. `weakness` — 0=None, 1=Mild, 2=Moderate, 3=Severe
4. `movement_limit` — 0=None, 1=Mild, 2=Moderate, 3=Severe
5. `pain_type` — 0=Dull, 1=Sharp, 2=Burning, 3=Throbbing
6. `duration` — 0=<24hrs, 1=1-3 days, 2=>3 days

**Output:** Risk category — Low (0), Moderate (1), High (2)

**Model:** Random Forest Classifier (200 trees) trained on 1,200 synthetic samples derived from clinical muscle injury symptom patterns (informed by EMG/PhysioNet literature).

---

## Expected Impact

- Reduces cost of preliminary screening from ₹5,000–20,000 → **₹0**
- Reaches rural/Tier-2/3 users without specialist access
- Encourages early medical consultation
- Results in **under 2 minutes** vs. days for lab tests
- Extensible to other musculoskeletal conditions
