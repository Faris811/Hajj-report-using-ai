import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Hajj Report AI", page_icon="🕋", layout="wide")

DEMO_DATA = [
    ("lost item in tent", "Low"), ("استفسار عن موقع الحملة", "Low"),
    ("ازدحام بسيط عند البوابة", "Medium"), ("مشكلة في النظافة", "Medium"),
    ("lost elderly pilgrim", "High"), ("حاج تائه ولا يعرف موقع مخيمه", "High"),
    ("medical emergency unconscious pilgrim", "Critical"),
    ("حالة إغماء وصعوبة في التنفس", "Critical"),
    ("fire in camp", "Critical"), ("حريق في المخيم", "Critical"),
    ("stampede and severe crowding", "Critical"),
    ("ازدحام شديد وتدافع", "Critical"),
    ("broken air conditioner", "Medium"), ("انقطاع التكييف عن المخيم", "Medium"),
    ("water shortage", "High"), ("نقص مياه الشرب بشكل كبير", "High"),
]

URGENT = [
    "fire", "حريق", "unconscious", "unresponsive", "إغماء", "فاقد الوعي",
    "breathing", "تنفس", "heart", "قلب", "bleeding", "نزيف", "accident",
    "حادث", "stampede", "تدافع", "collapse", "انهيار", "critical", "حرج"
]
HIGH = [
    "lost", "تائه", "مفقود", "missing", "water shortage", "نقص مياه",
    "overcrowd", "ازدحام شديد", "security", "أمني", "danger", "خطر"
]

def normalize(text: str) -> str:
    text = str(text).strip().lower()
    return re.sub(r"\s+", " ", text)

def rule_priority(text: str):
    t = normalize(text)
    if any(k in t for k in URGENT):
        return "Critical", 0.95
    if any(k in t for k in HIGH):
        return "High", 0.85
    return None, 0.0

@st.cache_resource
def train_model():
    x = [normalize(row[0]) for row in DEMO_DATA]
    y = [row[1] for row in DEMO_DATA]
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    model.fit(x, y)
    return model

def predict(text, model):
    text = normalize(text)
    rule_label, rule_conf = rule_priority(text)
    probs = model.predict_proba([text])[0]
    labels = model.classes_
    idx = probs.argmax()
    ml_label, ml_conf = labels[idx], float(probs[idx])
    if rule_label:
        return rule_label, max(rule_conf, ml_conf), "Safety rule + ML"
    return ml_label, ml_conf, "ML model"

st.title("🕋 Hajj Report AI")
st.caption("AI-assisted classification of Hajj reports by priority")
model = train_model()

tab1, tab2, tab3 = st.tabs(["Classify Report", "Batch CSV", "About"])

with tab1:
    report = st.text_area("Report", placeholder="Example: حاج تائه منذ ساعة ولا يعرف موقع مخيمه...", height=150)
    if st.button("Analyze Report", type="primary", use_container_width=True):
        if not report.strip():
            st.warning("Please enter a report first.")
        else:
            label, confidence, method = predict(report, model)
            st.metric("Priority", label)
            st.progress(min(confidence, 1.0), text=f"Confidence: {confidence:.0%}")
            st.info(f"Decision method: {method}")

with tab2:
    st.write("Upload a CSV containing a report/text column. Optional: priority/label column.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        candidates = [c for c in df.columns if c.lower() in {"report", "text", "description", "بلاغ", "البلاغ"}]
        if not candidates:
            st.error("No report column found. Use: report, text, description, بلاغ, or البلاغ.")
        else:
            col = candidates[0]
            results = [predict(v, model) for v in df[col].fillna("")]
            df["predicted_priority"] = [r[0] for r in results]
            df["confidence"] = [r[1] for r in results]
            df["method"] = [r[2] for r in results]
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download Results",
                df.to_csv(index=False).encode("utf-8-sig"),
                "hajj_report_predictions.csv",
                "text/csv"
            )

with tab3:
    st.markdown("""
### What this version does
- Classifies reports into Low / Medium / High / Critical.
- Supports English and Arabic examples.
- Uses TF-IDF + Logistic Regression as the baseline ML classifier.
- Adds explicit safety rules for high-risk terms.
- Supports single-report and CSV batch classification.
- Shows confidence and decision method.
- Keeps the original main branch untouched.

### Important
This is an AI-assisted prioritization tool, not a replacement for emergency or security dispatch decisions.
For production, train and validate the model on real, labeled historical reports.
""")
