# Hajj Report AI

AI-assisted classification of Hajj reports by priority.

## Improved version

This branch adds a complete Streamlit application for classifying reports as Low, Medium, High, or Critical.

## Features

- Arabic + English demonstration data
- TF-IDF + Logistic Regression baseline model
- Explicit high-risk safety rules
- Single-report classification
- Batch CSV classification
- Confidence score
- Downloadable prediction results
- Clear project documentation

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

## CSV format

The uploader accepts a report column named: report, text, description, بلاغ, or البلاغ.

## Architecture

Input report -> normalization -> safety-rule check + ML classifier -> priority + confidence -> UI/export

## Note

The included examples are demonstration data. Production use requires properly labeled historical Hajj reports, validation, human review, and appropriate safety controls.
