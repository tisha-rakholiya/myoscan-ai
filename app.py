
# MyoScan AI - Flask Web App
# Team: Code Catalyst
# Tisha Rakholiya, Shreya Nath

from flask import Flask, render_template, request
import pickle
import numpy as np
from fpdf import FPDF
import io
from flask import send_file

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

@app.route("/download-report", methods=["POST"])
def download_report():
    name       = request.form.get("name", "Patient")
    location   = request.form.get("location", "N/A")
    risk_level = request.form.get("risk_level", "N/A")
    confidence = request.form.get("confidence", "N/A")
    advice     = request.form.get("advice", "N/A")

    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(2, 128, 144)
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 10)
    pdf.cell(0, 10, "MyoScan AI - Risk Assessment Report", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_xy(10, 22)
    pdf.cell(0, 8, "AI-Assisted Muscle Injury Screening Tool", ln=True)

    # Reset color
    pdf.set_text_color(30, 42, 42)
    pdf.set_xy(10, 45)

    # Patient Info
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Patient Information", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Name: {name}", ln=True)
    pdf.cell(0, 7, f"Affected Location: {location}", ln=True)
    pdf.ln(5)

    # Risk Result
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Assessment Result", ln=True)
    pdf.set_font("Helvetica", "B", 18)
    colors = {"LOW": (2,195,154), "MODERATE": (245,158,11), "HIGH": (239,68,68)}
    c = colors.get(risk_level, (0,0,0))
    pdf.set_text_color(c[0], c[1], c[2])
    pdf.cell(0, 10, f"{risk_level} RISK  ({confidence}% confidence)", ln=True)
    pdf.set_text_color(30, 42, 42)
    pdf.ln(3)

    # Advice
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Medical Advice", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, advice)
    pdf.ln(5)

    # Disclaimer
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 6, "DISCLAIMER: This report is for preliminary screening only and is NOT a substitute for professional medical diagnosis. Always consult a qualified doctor.")
    pdf.ln(3)
    pdf.set_text_color(30, 42, 42)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, "Generated by MyoScan AI | Team Code Catalyst | Bharat Academix CodeQuest 2026", ln=True)

    # Send PDF
    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True,
                     download_name=f"MyoScan_Report_{name}.pdf")

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print("Starting MyoScan AI...")
    print(f"Running on port {port}")

    app.run(host="0.0.0.0", port=port)