# 🫀 HeartLink: Spec-Driven Clinical Guardrails

**Track:** AI Agents & Public Sector Impact (NHS Excellence)

**Vision:** Transforming raw clinical research into production-ready diagnostic tools using Agentic Engineering and AWS Kiro.

## 🚀 The Vision

In clinical settings, the journey from a data science model to a deployed tool is often blocked by a "Data Integrity Gap". HeartLink uses an agent-first approach to automate the transition from academic hypothesis to an executable, spec-driven clinical prototype that prioritizes reproducibility and patient safety.

## 🛑 The Problem: The Clinical "Noise" Gap

Standard diagnostic models often fail when they encounter "biologically impossible" data common in fragmented health records.

- **Medical Noise:** Our analysis of the UCI Heart Disease dataset identified critical outliers, including 172 cases of 0 cholesterol and 1 case of 0 blood pressure.
- **Data Integrity:** Traditional models trained on this noise become distorted, leading to inaccurate patient risk assessments.
- **The Deployment Hurdle:** Public sector projects require a "Spec-First" approach to ensure traceability and compliance with NHS standards.

## 🤖 The Agentic Approach

We leveraged AWS Kiro and Agentic Workflows to build a system that acts as both a developer and a clinical auditor.

### 1. Spec-First Engineering (EARS)

Using AWS Kiro, the agent generated unambiguous requirements using EARS (Easy Approach to Requirements Syntax):

> WHEN a patient record contains a vital sign of 0, THE SYSTEM SHALL flag the entry for manual review before model ingestion.

### 2. Clinical Guardrails

The agent implemented custom "Steering Files" to enforce domain-specific logic:

- **Median Imputation:** Automatically applied to vital signs to prevent outlier skew while maintaining central tendency.
- **Binary Classification:** Transformed the multi-class `num` field into a binary target ($0 = \text{Normal}$, $>0 = \text{Disease}$) to improve screening utility.

## 📊 Engineering Benchmarks

HeartLink is powered by a multi-layered diagnostic engine optimized through agentic iteration:

| Capability | Model | Performance / Insight |
|---|---|---|
| Risk Prediction | Random Forest | Achieved 84% Accuracy and 0.91 AUC. |
| Patient Stratification | K-Means | Identified 4 distinct profiles, including an "Asymptomatic High Risk" group. |
| Severity Analysis | Linear/Poly Reg | Achieved $R^2 \approx 0.5950$ by isolating high-quality linear features. |

## 🛠️ The "Zero to Shipped" Workflow

The repository provides a fully reproducible, end-to-end clinical pipeline:

1. **Ingest:** Loads the raw `heart_disease_dataset.csv`.
2. **Clean:** Executes agent-driven outlier detection and median-based imputation.
3. **Validate:** Runs 5-fold cross-validation to ensure model stability across clinical cohorts.
4. **Prescribe:** Maps model outputs to personalized medical guidance based on risk clusters.

## 🇬🇧 Public Sector Impact

By utilizing AWS Kiro, HeartLink demonstrates a path forward for Engineering Excellence in the NHS.

- **Traceability:** Every line of code is mapped back to a clinical requirement.
- **Safety:** The agent automatically blocks biologically impossible data from reaching the predictive engine.
- **Efficiency:** Accelerates the development of life-saving screening tools without compromising on data integrity or compliance.
