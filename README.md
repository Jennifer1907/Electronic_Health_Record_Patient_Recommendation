
# Preference-Aware Clinical Decision Support System (CDS)

## Overview
This project develops a preference-aware Clinical Decision Support (CDS) system designed to assist physicians and patients in making personalized treatment decisions.

The system integrates causal machine learning models with patient preference weighting to generate transparent and patient-centered treatment recommendations.

Traditional clinical decision tools rely on average treatment effects, which may not reflect the needs of individual patients. This system estimates individualized treatment effects (CATE) and combines them with patient priorities to support shared decision making (SDM) in clinical practice.

---

## Problem Statement

Medical treatment decisions often involve trade-offs across multiple outcomes such as:

- Pain reduction
- Functional improvement
- Rehabilitation time
- Out-of-pocket cost

However:

- Patients value these outcomes differently.
- Treatment benefits vary across individuals.
- Physicians need interpretable decision support tools.

Research Question:

How can predicted treatment outcomes and patient preferences be transparently combined to support shared decision making in clinical practice?

---
## Methodology

### 1. Data Exploration and Preprocessing

Key findings from exploratory analysis:

- Treatment assignment imbalance (~24% surgery)
- Outcome distributions are skewed
- Missing values present in clinical features
- Outcomes measured in different units

Solutions implemented:

- Propensity score weighting
- Missing data imputation
- Feature engineering
- Outcome normalization

---

## Causal Modeling Pipeline

The pipeline estimates patient-specific treatment effects using causal machine learning methods.

Steps:

1. Data preprocessing and feature engineering
2. Propensity score estimation
3. Covariate balance diagnostics
4. Causal meta-learner training
5. Model evaluation and selection

---

## Causal Learning Models

We evaluate several meta-learning approaches:

- S-Learner
- T-Learner
- X-Learner
- R-Learner
- Neural network meta-learners

Best Model:

S-Learner with XGBoost produced the most stable individualized treatment effect estimates.

---

## Preference-Aware Decision Framework

### Step 1 – Estimate Treatment Effects

Predict outcomes for:

- Surgery
- Conservative care

### Step 2 – Normalize Outcomes

Convert outcomes to a common dollar-value scale using willingness-to-pay anchors.

Example anchors:

- Pain improvement: $25,000
- Function improvement: $25,000
- Treatment-related pain: $15,000

### Step 3 – Apply Patient Preferences

Utility = Σ (Outcome_Value × Preference_Weight)

### Step 4 – Compute Utility Difference

ΔUtility = Utility(surgery) – Utility(conservative)

Decision rule:

- ΔUtility > 0 → Surgery recommended
- ΔUtility ≤ 0 → Conservative treatment recommended

---

## Sensitivity Analysis

The system tests whether recommendations remain stable when patient preferences change.

Procedure:

- Increase each preference weight by +20%
- Recompute the recommendation
- Flag cases where recommendations change

This identifies preference-sensitive decisions.

---

## Key Results

Treatment Recommendations:

- ~72% of patients recommended conservative care
- Many cases fall near the decision boundary
- High-confidence recommendations are rare

This highlights the importance of shared decision making.

---

## Clinical Heterogeneity

Treatment effects vary widely across patients.

Outcome | Treatment Effect Range
Pain reduction | -0.01 to +0.53
Functional improvement | -0.01 to +0.53
Rehabilitation time | +3.2 to +8.7 weeks
Cost impact | +$10k to +$30k

This confirms that one treatment does not fit all patients.

---

## Prototype System

A prototype CDS dashboard demonstrates clinical usage.

Features:

- Patient-specific treatment recommendation
- Outcome comparison visualization
- Confidence level indicator
- Sensitivity analysis results
- Preference contribution breakdown

The interface is designed for integration with Electronic Health Record (EHR) systems.

---

## Clinical Workflow

1. Retrieve predicted patient outcomes
2. Apply patient preference weights
3. Compare treatment options
4. Discuss trade-offs with the patient
5. Confirm final treatment decision

The system supports physician judgment rather than replacing it.

---

## Technologies Used

- Python
- XGBoost
- Scikit-learn
- Pandas
- Causal Machine Learning
- Jupyter Notebook
- Streamlit / Dashboard prototype

---

## Future Improvements

- Integration with hospital EHR systems
- Real-time clinical deployment
- Explainable AI visualization
- Larger multi-center clinical datasets

---

## Contributors

Team HAC
