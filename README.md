# Hajj Report AI

AI-assisted classification of Hajj reports by priority.

## Streamlit Community Cloud

This repository is ready to deploy on Streamlit Community Cloud.

1. Open Streamlit Community Cloud and sign in with GitHub.
2. Create a new app and select this repository.
3. Select branch **ai-enhanced** (or **main** after the pull request is merged).
4. Set the main file to **app.py**.
5. Click **Deploy**.

The dependencies are declared in `requirements.txt`, so Streamlit Cloud can install them automatically.

## Features

- Arabic + English demonstration data
- TF-IDF + Logistic Regression baseline model
- Explicit high-risk safety rules
- Single-report classification
- Batch CSV classification
- Confidence score
- Downloadable prediction results
- Streamlit Cloud deployment configuration

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

## CSV format

The uploader accepts a report column named: `report`, `text`, `description`, `بلاغ`, or `البلاغ`.

## Architecture

Input report -> normalization -> safety-rule check + ML classifier -> priority + confidence -> UI/export

## Important limitation

The included examples are demonstration data. This is a baseline/demo classifier, not a production Hajj decision system. Production use requires properly labeled historical Hajj reports, train/validation/test evaluation, human review, monitoring, and appropriate safety controls.
