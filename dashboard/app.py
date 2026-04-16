"""
HeartLink Dashboard — NHS-styled web interface for pipeline outputs.

Serves the HeartLink pipeline results (plots, metrics, reports) through
a Flask application styled with the NHS.UK frontend design system.

Usage::

    cd dashboard
    python app.py

Then open http://localhost:5000
"""

import csv
import os
import sys

from flask import Flask, render_template, request, send_from_directory

# Resolve paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

app = Flask(__name__, static_folder="static", template_folder="templates")


def _read_classification_report() -> str:
    """Read the classification report text file."""
    path = os.path.join(OUTPUTS_DIR, "classification_report.txt")
    if os.path.isfile(path):
        with open(path) as f:
            return f.read()
    return "Classification report not found."


def _read_regression_csv() -> list[dict]:
    """Read the regression variant comparison CSV."""
    path = os.path.join(OUTPUTS_DIR, "regression_variant_comparison.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _get_available_plots() -> dict[str, list[str]]:
    """Scan outputs/ and group available PNG files by category."""
    if not os.path.isdir(OUTPUTS_DIR):
        return {}

    files = sorted(os.listdir(OUTPUTS_DIR))
    pngs = [f for f in files if f.endswith(".png")]

    groups = {
        "eda": [],
        "classification": [],
        "regression": [],
        "clustering": [],
    }

    for f in pngs:
        if any(f.startswith(p) for p in [
            "correlation_", "target_distribution", "distribution_",
            "boxplot_", "cat_distribution_", "target_chol",
        ]):
            groups["eda"].append(f)
        elif any(f.startswith(p) for p in [
            "confusion_", "roc_", "feature_importance",
            "classifier_performance", "classification_",
        ]):
            groups["classification"].append(f)
        elif f.startswith("regression_"):
            groups["regression"].append(f)
        elif f.startswith("clustering_"):
            groups["clustering"].append(f)

    return groups


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Landing page — choose between model dashboard and user journeys."""
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Model dashboard routes (extend base_model.html)
# ---------------------------------------------------------------------------

@app.route("/model")
def model_index():
    """Model overview — key metrics, plots, regression table."""
    plots = _get_available_plots()
    report = _read_classification_report()
    variants = _read_regression_csv()
    return render_template(
        "model_index.html",
        plots=plots,
        report=report,
        variants=variants,
    )


@app.route("/model/eda")
def eda():
    """Exploratory Data Analysis page."""
    plots = _get_available_plots()
    return render_template("eda.html", plots=plots.get("eda", []))


@app.route("/model/classification")
def classification():
    """Classification results page."""
    plots = _get_available_plots()
    report = _read_classification_report()
    return render_template(
        "classification.html",
        plots=plots.get("classification", []),
        report=report,
    )


@app.route("/model/regression")
def regression():
    """Regression analysis page."""
    plots = _get_available_plots()
    variants = _read_regression_csv()
    return render_template(
        "regression.html",
        plots=plots.get("regression", []),
        variants=variants,
    )


@app.route("/model/clustering")
def clustering():
    """Clustering analysis page."""
    plots = _get_available_plots()
    return render_template(
        "clustering.html",
        plots=plots.get("clustering", []),
    )


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """Serve files from the pipeline outputs/ directory."""
    return send_from_directory(OUTPUTS_DIR, filename)


# ---------------------------------------------------------------------------
# User journeys routes (extend base_users.html)
# ---------------------------------------------------------------------------

@app.route("/users")
def users_index():
    """User journeys landing page."""
    return render_template("users_index.html")


@app.route("/users/patient", methods=["GET", "POST"])
def patient():
    """Patient screening form — enter vitals, get risk assessment."""
    result = None
    if request.method == "POST":
        try:
            age = int(request.form.get("age", 0))
            sex = int(request.form.get("sex", 0))
            cp = int(request.form.get("cp", 0))
            trestbps = int(request.form.get("trestbps", 0))
            chol = int(request.form.get("chol", 0))
            fbs = int(request.form.get("fbs", 0))
            restecg = int(request.form.get("restecg", 0))
            thalch = int(request.form.get("thalch", 0))
            exang = int(request.form.get("exang", 0))
            oldpeak = float(request.form.get("oldpeak", 0))
            slope = int(request.form.get("slope", 0))
        except (ValueError, TypeError):
            result = {
                "risk": "Error",
                "cluster": "N/A",
                "detail": "Please check your inputs — some values could not be read.",
                "colour": "red",
            }
            return render_template("patient.html", result=result, active_page="patient")

        # Simple mock risk logic
        risk_score = 0
        if age > 60:
            risk_score += 2
        elif age > 50:
            risk_score += 1
        if chol > 250:
            risk_score += 2
        elif chol > 200:
            risk_score += 1
        if trestbps > 140:
            risk_score += 1
        if cp in (1, 2, 3):
            risk_score += 1
        if exang == 1:
            risk_score += 1
        if oldpeak > 2.0:
            risk_score += 1
        if fbs == 1:
            risk_score += 1

        if risk_score >= 5:
            result = {
                "risk": "Elevated",
                "cluster": "Cluster 3 — High Risk",
                "detail": "Based on your inputs, your risk profile is elevated. "
                          "Please discuss these results with your GP as soon as possible.",
                "colour": "red",
            }
        elif risk_score >= 3:
            result = {
                "risk": "Moderate",
                "cluster": "Cluster 2 — Potential Risk",
                "detail": "Some of your values suggest moderate risk. "
                          "A follow-up appointment with your GP is recommended.",
                "colour": "yellow",
            }
        else:
            result = {
                "risk": "Low",
                "cluster": "Cluster 1 — Low Risk",
                "detail": "Your screening results suggest low risk. "
                          "Continue with regular check-ups and a healthy lifestyle.",
                "colour": "green",
            }

    return render_template("patient.html", result=result, active_page="patient")


@app.route("/users/clinician")
def clinician():
    """Clinician triage dashboard — mock patient case with data quality flags."""
    report = _read_classification_report()
    return render_template("clinician.html", report=report, active_page="clinician")


@app.route("/users/clinical-lead")
def clinical_lead():
    """Clinical Lead — service performance overview with real pipeline metrics."""
    report = _read_classification_report()
    return render_template("clinical_lead.html", report=report, active_page="clinical-lead")


@app.route("/users/radiographer")
def radiographer():
    """Radiographer — data quality feedback for high-missingness variables."""
    return render_template("radiographer.html", active_page="radiographer")


@app.route("/users/pharmacist", methods=["GET", "POST"])
def pharmacist():
    """Pharmacist — NHS BP monitoring service with NICE NG136 staging."""
    result = None
    if request.method == "POST":
        try:
            age = int(request.form.get("age", 0))
            systolic = int(request.form.get("systolic", 0))
            diastolic = int(request.form.get("diastolic", 0))
            chol = int(request.form.get("chol", 0)) if request.form.get("chol") else 0
        except (ValueError, TypeError):
            result = {
                "age": "—", "systolic": "—", "diastolic": "—", "chol": "—",
                "bp_stage": "error", "bp_stage_label": "Error",
                "heartlink": None,
            }
            return render_template("pharmacist.html", result=result, active_page="pharmacist")

        # NICE NG136 BP staging (clinic readings)
        if systolic >= 180 or diastolic >= 120:
            bp_stage = "severe"
            bp_stage_label = "Severe hypertension"
        elif systolic >= 160 or diastolic >= 100:
            bp_stage = "stage2"
            bp_stage_label = "Stage 2 hypertension"
        elif systolic >= 140 or diastolic >= 90:
            bp_stage = "stage1"
            bp_stage_label = "Stage 1 hypertension"
        else:
            bp_stage = "normal"
            bp_stage_label = "Normal"

        # HeartLink cardiovascular risk (simple mock using BP, age, chol)
        heartlink = None
        if chol > 0:
            risk_score = 0
            if age > 60:
                risk_score += 2
            elif age > 50:
                risk_score += 1
            if chol > 250:
                risk_score += 2
            elif chol > 200:
                risk_score += 1
            if systolic > 140:
                risk_score += 2
            elif systolic > 130:
                risk_score += 1

            if risk_score >= 4:
                heartlink = {
                    "risk": "Elevated",
                    "cluster": "Cluster 3 — High Risk",
                    "advice": "Multiple elevated cardiovascular markers. Urgent GP referral recommended.",
                    "colour": "red",
                }
            elif risk_score >= 2:
                heartlink = {
                    "risk": "Potential Risk",
                    "cluster": "Cluster 2 — Potential Risk",
                    "advice": "Offer lifestyle advice. Book routine GP follow-up within 2 weeks.",
                    "colour": "yellow",
                }
            else:
                heartlink = {
                    "risk": "Low",
                    "cluster": "Cluster 1 — Low Risk",
                    "advice": "Reassure patient. Recommend annual check-up.",
                    "colour": "green",
                }

        result = {
            "age": age,
            "systolic": systolic,
            "diastolic": diastolic,
            "chol": chol if chol > 0 else "Not recorded",
            "bp_stage": bp_stage,
            "bp_stage_label": bp_stage_label,
            "heartlink": heartlink,
        }

    return render_template("pharmacist.html", result=result, active_page="pharmacist")


@app.route("/users/workforce")
def workforce():
    """Workforce Leader — strategic overview of screening capacity and training needs."""
    return render_template("workforce.html", active_page="workforce")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
