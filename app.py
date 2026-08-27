import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MovieHit AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 10% 10%, rgba(124, 58, 237, 0.16), transparent 30%),
                radial-gradient(circle at 90% 20%, rgba(59, 130, 246, 0.12), transparent 30%),
                #09090f;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* Hero Section */
.hero {
    padding: 40px;
    border-radius: 20px;
    margin-bottom: 25px;
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.18), rgba(30, 41, 59, 0.35));
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin-bottom: 8px;
    background: linear-gradient(90deg, #ffffff, #c4b5fd, #93c5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 16px;
    color: #a1a1aa;
    max-width: 650px;
    line-height: 1.5;
}

.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(124, 58, 237, 0.15);
    border: 1px solid rgba(167, 139, 250, 0.25);
    color: #c4b5fd;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 14px;
}

/* Section titles */
.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #ffffff;
    margin-top: 10px;
}

.section-description {
    color: #a1a1aa;
    font-size: 14px;
    margin-bottom: 20px;
}

/* Prediction Result Cards */
.prediction-hit {
    padding: 24px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(20, 83, 45, 0.25));
    border: 1px solid rgba(74, 222, 128, 0.3);
    margin-bottom: 20px;
}

.prediction-hit h2 {
    color: #4ade80;
    margin: 0 0 8px 0;
    font-size: 22px;
}

.prediction-hit p {
    color: #e2e8f0;
    margin: 0;
    font-size: 14px;
}

.prediction-nohit {
    padding: 24px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(127, 29, 29, 0.25));
    border: 1px solid rgba(248, 113, 113, 0.3);
    margin-bottom: 20px;
}

.prediction-nohit h2 {
    color: #f87171;
    margin: 0 0 8px 0;
    font-size: 22px;
}

.prediction-nohit p {
    color: #e2e8f0;
    margin: 0;
    font-size: 14px;
}

/* Cards & Metrics */
.metric-card {
    background: rgba(24, 24, 32, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 18px;
    text-align: center;
}

.metric-value {
    font-size: 24px;
    font-weight: 800;
    color: #ffffff;
}

.metric-label {
    font-size: 12px;
    color: #a1a1aa;
    margin-top: 4px;
    font-weight: 600;
}

/* Footer */
.footer {
    text-align: center;
    color: #71717a;
    font-size: 13px;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}

.custom-divider {
    height: 1px;
    background: rgba(255, 255, 255, 0.08);
    margin: 30px 0;
}

/* Streamlit Native Elements Customization */
div[data-testid="stForm"] {
    border: 1px solid rgba(255, 255, 255, 0.08);
    background-color: rgba(24, 24, 32, 0.5);
    border-radius: 16px;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid rgba(167, 139, 250, 0.3);
    padding: 12px 20px;
    font-size: 16px;
    font-weight: 700;
    color: white;
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #a78bfa;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

model_package = joblib.load("models/movie_hit_model.pkl")
model = model_package["model"]
model_name = model_package["model_name"]
genres = model_package["genres"]

# =========================================================
# HERO SECTION
# =========================================================

hero_html = """
<div class="hero">
    <div class="badge">✦ MACHINE LEARNING • BOX OFFICE PREDICTION</div>
    <div class="hero-title">🎬 MovieHit AI</div>
    <div class="hero-subtitle">
        Discover whether a movie has the characteristics of a potential commercial hit — before it reaches the big screen.
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# =========================================================
# INPUT SECTION
# =========================================================

st.markdown('<div class="section-title">🎥 Build Your Movie</div>', unsafe_allow_html=True)
st.markdown('<div class="section-description">Enter the production details below to generate an AI prediction.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    budget = st.number_input(
        "💰 Production Budget ($)",
        min_value=10000.0,
        max_value=500000000.0,
        value=50000000.0,
        step=1000000.0,
        help="Estimated production budget of the movie."
    )

    runtime = st.number_input(
        "⏱️ Runtime (minutes)",
        min_value=1,
        max_value=300,
        value=120
    )

    release_year = st.number_input(
        "📅 Release Year",
        min_value=1900,
        max_value=2030,
        value=2026
    )

    release_month = st.selectbox(
        "🗓️ Release Month",
        options=list(range(1, 13)),
        format_func=lambda x: pd.Timestamp(year=2020, month=x, day=1).strftime("%B")
    )

with col2:
    main_genre = st.selectbox(
        "🎭 Main Genre",
        genres
    )

    num_production_companies = st.number_input(
        "🏢 Production Companies",
        min_value=0,
        max_value=20,
        value=2
    )

    num_production_countries = st.number_input(
        "🌍 Production Countries",
        min_value=0,
        max_value=20,
        value=1
    )

    num_cast = st.number_input(
        "👥 Cast Members",
        min_value=0,
        max_value=200,
        value=10
    )

    num_crew = st.number_input(
        "🎞️ Crew Members",
        min_value=0,
        max_value=1000,
        value=50
    )

# =========================================================
# PREDICT BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("🔮 PREDICT MOVIE SUCCESS", use_container_width=True)

# =========================================================
# PREDICTION RESULTS
# =========================================================

if predict:
    log_budget = np.log1p(budget)
    budget_per_minute = budget / runtime

    input_data = pd.DataFrame({
        "budget": [budget],
        "runtime": [runtime],
        "release_year": [release_year],
        "release_month": [release_month],
        "main_genre": [main_genre],
        "num_production_companies": [num_production_companies],
        "num_production_countries": [num_production_countries],
        "num_cast": [num_cast],
        "num_crew": [num_crew],
        "log_budget": [log_budget],
        "budget_per_minute": [budget_per_minute]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    hit_probability = probability[1] * 100
    not_hit_probability = probability[0] * 100

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 AI Prediction</div>', unsafe_allow_html=True)

    if prediction == 1:
        hit_card = """
        <div class="prediction-hit">
            <h2>🔥 LIKELY TO BE A HIT</h2>
            <p>Based on the information provided, MovieHit AI predicts that this movie has characteristics associated with commercial success.</p>
        </div>
        """
        st.markdown(hit_card, unsafe_allow_html=True)
    else:
        nohit_card = """
        <div class="prediction-nohit">
            <h2>📉 LIKELY NOT TO BE A HIT</h2>
            <p>Based on the information provided, MovieHit AI predicts a lower likelihood of commercial success.</p>
        </div>
        """
        st.markdown(nohit_card, unsafe_allow_html=True)

    # Probabilities
    probability_col1, probability_col2 = st.columns(2)

    with probability_col1:
        card_1 = f"""
        <div class="metric-card">
            <div class="metric-value">{hit_probability:.1f}%</div>
            <div class="metric-label">HIT PROBABILITY</div>
        </div>
        """
        st.markdown(card_1, unsafe_allow_html=True)

    with probability_col2:
        card_2 = f"""
        <div class="metric-card">
            <div class="metric-value">{not_hit_probability:.1f}%</div>
            <div class="metric-label">NOT-HIT PROBABILITY</div>
        </div>
        """
        st.markdown(card_2, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(float(probability[1]))

    # Model Overview
    st.markdown("<br>", unsafe_allow_html=True)
    info1, info2, info3 = st.columns(3)

    with info1:
        info_card1 = f"""
        <div class="metric-card">
            <div style="font-size:22px;">🌲</div>
            <div class="metric-value" style="font-size:18px;">{model_name}</div>
            <div class="metric-label">FINAL MODEL</div>
        </div>
        """
        st.markdown(info_card1, unsafe_allow_html=True)

    with info2:
        info_card2 = """
        <div class="metric-card">
            <div style="font-size:22px;">🎯</div>
            <div class="metric-value" style="font-size:22px;">66.72%</div>
            <div class="metric-label">MODEL ACCURACY</div>
        </div>
        """
        st.markdown(info_card2, unsafe_allow_html=True)

    with info3:
        info_card3 = """
        <div class="metric-card">
            <div style="font-size:22px;">📈</div>
            <div class="metric-value" style="font-size:22px;">63.87%</div>
            <div class="metric-label">F1 SCORE</div>
        </div>
        """
        st.markdown(info_card3, unsafe_allow_html=True)

    # Summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎞️ Movie Profile</div>', unsafe_allow_html=True)

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.write(f"**Genre:** {main_genre}")
        st.write(f"**Budget:** ${budget:,.0f}")
        st.write(f"**Runtime:** {runtime} minutes")

    with summary_col2:
        release_date_str = pd.Timestamp(year=int(release_year), month=int(release_month), day=1).strftime('%B %Y')
        st.write(f"**Release:** {release_date_str}")
        st.write(f"**Production Companies:** {num_production_companies}")
        st.write(f"**Cast / Crew:** {num_cast} / {num_crew}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 This prediction is a machine learning estimate based on historical movie data. It should not be treated as a guarantee of actual box-office performance.")

# =========================================================
# MODEL PERFORMANCE SECTION
# =========================================================

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧠 About the Model</div>', unsafe_allow_html=True)
st.markdown('<div class="section-description">MovieHit AI compares multiple classification algorithms and selects the strongest model based on F1 score.</div>', unsafe_allow_html=True)

performance_cols = st.columns(4)
metrics = [
    ("🌲", model_name, "Final Model"),
    ("🎯", "66.72%", "Accuracy"),
    ("⚡", "69.85%", "Precision"),
    ("📈", "63.87%", "F1 Score")
]

for col, metric in zip(performance_cols, metrics):
    with col:
        perf_card = f"""
        <div class="metric-card">
            <div style="font-size:22px;">{metric[0]}</div>
            <div class="metric-value" style="font-size:18px;">{metric[1]}</div>
            <div class="metric-label">{metric[2]}</div>
        </div>
        """
        st.markdown(perf_card, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

footer_html = """
<div class="footer">
    🎬 MovieHit AI &nbsp;•&nbsp; End-to-End Machine Learning Project
    <br><br>
    Built with Python • Scikit-learn • Pandas • Streamlit
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)