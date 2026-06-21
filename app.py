
# MyoScan AI - Flask Web App
# Team: Code Catalyst
# Tisha Rakholiya, Shreya Nath

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load trained model
model = pickle.load(open("model/model.pkl", "rb"))
print("Model loaded successfully")


@app.route("/scan")
def scan():
    return render_template("scan.html")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():

    # read form inputs
    name = request.form.get("name", "").strip()
    location = request.form.get("location", "Not specified")

    pain = int(request.form["pain_intensity"])
    swell = int(request.form["swelling"])
    weak = int(request.form["weakness"])
    move = int(request.form["movement_limit"])
    ptype = int(request.form["pain_type"])
    dur = int(request.form["duration"])

    age = int(request.form.get("age", 30))
    prev = int(request.form.get("previous_injury", 0))
    actv = int(request.form.get("activity_level", 1))

    # predict using model
    features = np.array([
        [pain, swell, weak, move, ptype, dur, age, prev, actv]
    ])

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    confidence = round(prob[pred] * 100, 1)

    low_prob = round(prob[0] * 100, 1)
    mod_prob = round(prob[1] * 100, 1)
    high_prob = round(prob[2] * 100, 1)

    # feature importance
    feat_names = [
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

    importances = model.feature_importances_

    feat_imp = sorted(
        zip(feat_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    top_factors = [
        (name, round(score * 100, 1))
        for name, score in feat_imp[:3]
    ]

    # decide output based on prediction
    if pred == 0:

        risk_level = "LOW"
        risk_color = "#02C39A"
        risk_bg = "#E8FAF7"
        emoji = "🟢"

        message = (
            "Your symptoms suggest a LOW risk of significant muscle injury."
        )

        advice = (
            "Rest and monitor. Medical visit is likely not urgent "
            "— but see a doctor if symptoms worsen."
        )

        steps = [
            "Rest the affected area for 24-48 hours",
            "Apply ice pack for 15-20 minutes every few hours",
            "Avoid activities that trigger the pain",
            "Take OTC pain relief if needed",
            "Monitor symptoms carefully"
        ]

    elif pred == 1:

        risk_level = "MODERATE"
        risk_color = "#F59E0B"
        risk_bg = "#FFFBEB"
        emoji = "🟡"

        message = (
            "Your symptoms suggest a MODERATE risk of muscle injury."
        )

        advice = (
            "We recommend seeing a doctor within 2-3 days "
            "for a proper evaluation."
        )

        steps = [
            "Avoid strenuous activity",
            "Use RICE therapy",
            "Book a doctor or physiotherapy appointment",
            "Monitor swelling and pain",
            "Use anti-inflammatory medication if needed"
        ]

    else:

        risk_level = "HIGH"
        risk_color = "#EF4444"
        risk_bg = "#FEF2F2"
        emoji = "🔴"

        message = (
            "Your symptoms suggest a HIGH risk of significant muscle injury."
        )

        advice = (
            "Please seek medical attention as soon as possible."
        )

        steps = [
            "Visit a clinic or hospital immediately",
            "Immobilize the affected area",
            "Use ice, avoid heat",
            "Avoid painful movement",
            "Seek emergency care if symptoms are severe"
        ]

    # Explainability
    reasons = []

    if pain >= 8:
        reasons.append("Severe pain intensity detected")

    if weak >= 2:
        reasons.append("Moderate to severe muscle weakness reported")

    if move >= 2:
        reasons.append("Significant movement restriction observed")

    if swell >= 2:
        reasons.append("Noticeable swelling detected")

    if dur >= 1:
        reasons.append("Symptoms have persisted for multiple days")

    if prev == 1:
        reasons.append("Previous injury history increases risk")

    if actv >= 2:
        reasons.append("High physical activity level detected")

    return render_template(
        "result.html",
        name=name,
        location=location,
        risk_level=risk_level,
        risk_color=risk_color,
        risk_bg=risk_bg,
        emoji=emoji,
        message=message,
        advice=advice,
        steps=steps,
        confidence=confidence,
        low_prob=low_prob,
        mod_prob=mod_prob,
        high_prob=high_prob,
        top_factors=top_factors,
        reasons=reasons
    )


if __name__ == "__main__":
    print("Starting MyoScan AI...")
    print("Open browser -> http://127.0.0.1:5000")
    app.run(debug=True)

