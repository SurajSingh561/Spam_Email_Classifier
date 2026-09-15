import re
import pickle
import streamlit as st
import plotly.graph_objects as go

# =====================================
# Spam Keywords
# =====================================

spam_words = [
    "free",
    "winner",
    "won",
    "prize",
    "offer",
    "claim",
    "click",
    "urgent",
    "cash",
    "bonus",
    "gift",
    "lottery",
    "otp",
    "reward",
    "limited",
    "congratulations",
]

# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="AI Spam Email Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================
# Load CSS
# =====================================

with open("styles/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =====================================
# Load ML Model
# =====================================

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# =====================================
# Sidebar
# =====================================

with st.sidebar:

    st.title("🤖 AI Spam Detector")

    st.markdown("---")

    st.success("Model Loaded Successfully")

    st.metric("Accuracy", "96.23%")
    st.metric("Algorithm", "Multinomial Naive Bayes")
    st.metric("Vectorizer", "TF-IDF")

    st.markdown("---")

    st.info(
        """
This application classifies Email and SMS messages
as Spam or Not Spam using Machine Learning.
"""
    )

# =====================================
# Hero Section
# =====================================

st.markdown(
    """
<div class="hero">

<h1>AI Spam Email Detector</h1>

<p>
Machine Learning powered Email & SMS Spam Detection
using TF-IDF Vectorization and Multinomial Naive Bayes.
</p>

<div class="hero-badges">
<span>Python</span>
<span>Streamlit</span>
<span>Scikit-learn</span>
<span>TF-IDF</span>
<span>Naive Bayes</span>
</div>

</div>
""",
    unsafe_allow_html=True,
)

# =====================================
# Main Layout
# =====================================

left, right = st.columns([2.2, 1])

# =====================================
# Left Panel
# =====================================

with left:

    st.subheader("✉️ Enter Email or SMS")

    message = st.text_area(
        label="",
        height=50,
        placeholder="Type your Email or SMS here..."
    )

    uploaded_file = st.file_uploader(
        "📂 Upload a .txt file (Optional)",
        type=["txt"]
    )

    predict = st.button(
        "🔍 Analyze Message",
        use_container_width=True
    )

# =====================================
# Right Panel
# =====================================

with right:

    st.markdown(
        """
<div class="info-card">

<h2>📊 Model Information</h2>

<div class="info-box">
<h4>Accuracy</h4>
<p>96.23%</p>
</div>

<div class="info-box">
<h4>Algorithm</h4>
<p>Multinomial Naive Bayes</p>
</div>

<div class="info-box">
<h4>Vectorizer</h4>
<p>TF-IDF</p>
</div>

</div>
""",
        unsafe_allow_html=True,
    )

# =====================================
# Prediction Starts Here
# =====================================

if predict:

    # Read uploaded file

    if uploaded_file is not None:
        message = uploaded_file.read().decode("utf-8")

    # Empty input

    if message.strip() == "":
        st.warning("⚠ Please enter a message or upload a text file.")
        st.stop()

    # ML Prediction

    transformed_message = vectorizer.transform([message])

    prediction = model.predict(transformed_message)

    probability = model.predict_proba(transformed_message)

    spam_prob = probability[0][1] * 100
    ham_prob = probability[0][0] * 100

    # Message Statistics

    word_count = len(message.split())
    character_count = len(message)
    uppercase_count = sum(1 for c in message if c.isupper())
    number_count = len(re.findall(r"\d", message))
    link_count = len(re.findall(r"http[s]?://|www\.", message))

    # =====================================
    # Prediction Result
    # =====================================

    st.divider()

    st.subheader("📩 Prediction Result")

    if prediction[0] == 1:

        st.markdown(
            f"""
            <div class="spam-card">
                <h2>🚨 SPAM DETECTED</h2>
                <p>Confidence: <b>{spam_prob:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(float(probability[0][1]))

    else:

        st.markdown(
            f"""
            <div class="ham-card">
                <h2>✅ SAFE MESSAGE</h2>
                <p>Confidence: <b>{ham_prob:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(float(probability[0][0]))

    # =====================================
    # Probability Chart
    # =====================================

    st.divider()

    st.subheader("📊 Prediction Confidence")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Ham", "Spam"],
            y=[ham_prob, spam_prob],
            text=[
                f"{ham_prob:.1f}%",
                f"{spam_prob:.1f}%"
            ],
            textposition="outside",
            marker_color=[
                "#22C55E",
                "#EF4444"
            ],
            marker_line_color="white",
            marker_line_width=2,
            width=0.45,
        )
    )

    fig.update_layout(

        title="Prediction Confidence",

        template="plotly_dark",

        height=460,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(size=15),

        xaxis_title="Category",

        yaxis_title="Confidence (%)",

        yaxis=dict(
            range=[0, 100],
            gridcolor="rgba(255,255,255,0.08)"
        ),

        margin=dict(
            l=30,
            r=30,
            t=60,
            b=30
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # =====================================
    # Message Statistics
    # =====================================

    st.divider()

    st.subheader("📈 Message Statistics")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Words", word_count)
    col2.metric("Characters", character_count)
    col3.metric("Uppercase", uppercase_count)
    col4.metric("Numbers", number_count)
    col5.metric("Links", link_count)

    # =====================================
    # Spam Keywords
    # =====================================

    st.divider()

    st.subheader("⚠️ Spam Keywords Detected")

    found_keywords = []

    for word in spam_words:
        if word in message.lower():
            found_keywords.append(word)

    if found_keywords:

        cols = st.columns(min(4, len(found_keywords)))

        for i, word in enumerate(found_keywords):
            cols[i % len(cols)].error(word.upper())

    else:

        st.success("✅ No suspicious keywords detected.")   
    

# =====================================
# Footer
# =====================================

st.markdown("---")

st.caption(
    "Developed by Suraj Singh | Python • Scikit-learn • Streamlit"
)        