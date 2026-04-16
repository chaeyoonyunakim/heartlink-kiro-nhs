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

from flask import Flask, render_template, send_from_directory

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


@app.route("/")
def index():
    """Dashboard home — overview with key metrics."""
    plots = _get_available_plots()
    report = _read_classification_report()
    variants = _read_regression_csv()
    return render_template(
        "index.html",
        plots=plots,
        report=report,
        variants=variants,
    )


@app.route("/eda")
def eda():
    """Exploratory Data Analysis page."""
    plots = _get_available_plots()
    return render_template("eda.html", plots=plots.get("eda", []))


@app.route("/classification")
def classification():
    """Classification results page."""
    plots = _get_available_plots()
    report = _read_classification_report()
    return render_template(
        "classification.html",
        plots=plots.get("classification", []),
        report=report,
    )


@app.route("/regression")
def regression():
    """Regression analysis page."""
    plots = _get_available_plots()
    variants = _read_regression_csv()
    return render_template(
        "regression.html",
        plots=plots.get("regression", []),
        variants=variants,
    )


@app.route("/clustering")
def clustering():
    """Clustering analysis page."""
    plots = _get_available_plots()
    return render_template(
        "clustering.html",
        plots=plots.get("clustering", []),
    )


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """Serve files from the pipeline outputs/ directory."""
    return send_from_directory(OUTPUTS_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
