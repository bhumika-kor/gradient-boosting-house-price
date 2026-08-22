# =========================================================
# HOUSEPREDICT AI
# HOUSE PRICE PREDICTION
# GRADIENT BOOSTING REGRESSOR
# =========================================================

import streamlit as st
import joblib
import pandas as pd


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HousePredict AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# LOAD MODEL
# =========================================================

try:
    model = joblib.load("best_house_price_model.pkl")
except FileNotFoundError:
    st.error("❌ best_house_price_model.pkl not found.")
    st.stop()


# =========================================================
# CSS
# =========================================================

st.html("""
<style>

/* =====================================================
   MAIN APP
   ===================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(157,78,221,0.12),
            transparent 25%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(123,44,191,0.10),
            transparent 25%
        ),
        #08080c;

    color: white;
}

.block-container {
    max-width: 1150px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}


/* =====================================================
   NAVBAR
   ===================================================== */

.navbar {
    background: rgba(15,15,23,0.98);
    border: 1px solid rgba(157,78,221,0.35);
    border-radius: 18px;
    padding: 16px 25px;
    margin-bottom: 25px;
}

.brand {
    font-size: 24px;
    font-weight: 800;
    color: #ffffff !important;
}

.brand span {
    color: #d8a4ff !important;
}


/* =====================================================
   TOP NAVIGATION BUTTONS
   ===================================================== */

.stRadio > div {
    gap: 10px;
}

.stRadio label {
    color: #ffffff !important;
    background-color: #17121f !important;
    border: 1px solid #5a189a !important;
    border-radius: 12px !important;
    padding: 8px 18px !important;
}

.stRadio label p {
    color: #ffffff !important;
    font-weight: 700 !important;
}

.stRadio label:hover {
    background-color: #2a1740 !important;
    border-color: #9d4edd !important;
}

.stRadio [data-checked="true"] {
    background-color: #5a189a !important;
    border-color: #c77dff !important;
}

.stRadio [data-checked="true"] p {
    color: #ffffff !important;
    font-weight: 800 !important;
}


/* =====================================================
   HERO
   ===================================================== */

.hero {
    background:
        radial-gradient(
            circle at 85% 30%,
            rgba(199,125,255,0.20),
            transparent 28%
        ),
        linear-gradient(
            135deg,
            #13001c,
            #240046 55%,
            #3c096c
        );

    border: 1px solid rgba(199,125,255,0.30);
    border-radius: 28px;
    padding: 50px;
    margin-bottom: 25px;
}

.hero-tag {
    display: inline-block;
    padding: 7px 15px;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    color: #e0aaff;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 50px;
    line-height: 1.1;
    font-weight: 800;
    color: #ffffff !important;
}

.hero-title span {
    color: #d8a4ff !important;
}

.hero-text {
    color: #e0d8e8 !important;
    font-size: 17px;
    line-height: 1.7;
    max-width: 750px;
}


/* =====================================================
   SECTION
   ===================================================== */

.section-heading {
    text-align: center;
    font-size: 31px;
    font-weight: 800;
    color: #ffffff !important;
    margin-top: 25px;
    margin-bottom: 8px;
}

.section-subtitle {
    text-align: center;
    color: #bdb4c7 !important;
    font-size: 14px;
    margin-bottom: 30px;
}


/* =====================================================
   CARDS
   ===================================================== */

.card {
    background: #111118;
    border: 1px solid rgba(157,78,221,0.30);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 20px;
}

.card-title {
    color: #d8a4ff !important;
    font-size: 21px;
    font-weight: 700;
}

.card-text {
    color: #c9c1d0 !important;
    font-size: 14px;
    line-height: 1.8;
}


/* =====================================================
   INPUT LABELS
   ===================================================== */

label {
    color: #ffffff !important;
    font-weight: 600 !important;
}

.stNumberInput label,
.stSelectbox label {
    color: #ffffff !important;
}


/* =====================================================
   NUMBER INPUT
   ===================================================== */

.stNumberInput input {
    background-color: #15151d !important;
    color: #ffffff !important;
    border: 1px solid #5a189a !important;
    border-radius: 11px !important;
}

.stNumberInput input:focus {
    border-color: #c77dff !important;
    color: #ffffff !important;
}


/* =====================================================
   SELECT BOX
   ===================================================== */

.stSelectbox div[data-baseweb="select"] {
    background-color: #15151d !important;
    color: #ffffff !important;
    border: 1px solid #5a189a !important;
    border-radius: 11px !important;
}


/* =====================================================
   BUTTON
   ===================================================== */

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 13px;

    background: linear-gradient(
        135deg,
        #5a189a,
        #7b2cbf,
        #9d4edd
    );

    color: #ffffff !important;
    font-size: 17px;
    font-weight: 700;
    border: none;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0px 8px 25px rgba(157,78,221,0.35);
}


/* =====================================================
   RESULT CARD
   ===================================================== */

.prediction-card {
    background:
        radial-gradient(
            circle at 50% 0%,
            rgba(224,170,255,0.18),
            transparent 45%
        ),
        linear-gradient(
            135deg,
            #16001f,
            #3c096c,
            #5a189a
        );

    border: 1px solid #a855d8;
    border-radius: 22px;
    padding: 35px;
    text-align: center;
    margin-top: 30px;
}

.prediction-title {
    color: #f0e8f5 !important;
    font-size: 19px;
    font-weight: 600;
}

.prediction-value {
    color: #f0c9ff !important;
    font-size: 43px;
    font-weight: 800;
}


/* =====================================================
   INFO CARDS
   ===================================================== */

.info-card {
    background: linear-gradient(
        135deg,
        #111118,
        #18121e
    );

    border: 1px solid rgba(157,78,221,0.30);
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    min-height: 150px;
}

.info-icon {
    font-size: 35px;
}

.info-title {
    color: #d8a4ff !important;
    font-weight: 700;
    font-size: 17px;
}

.info-text {
    color: #bcb3c5 !important;
    font-size: 13px;
    margin-top: 7px;
}


/* =====================================================
   CHAT
   ===================================================== */

.stChatInput textarea {
    background-color: #15151d !important;
    color: #ffffff !important;
}

.stChatInput textarea::placeholder {
    color: #aaa2b0 !important;
}


/* =====================================================
   FOOTER
   ===================================================== */

.footer {
    text-align: center;
    color: #aaa2b0 !important;
    font-size: 13px;
    margin-top: 45px;
    padding-top: 25px;
    border-top: 1px solid #24212a;
}

</style>
""")


# =========================================================
# NAVBAR
# =========================================================

st.html("""
<div class="navbar">
    <div class="brand">
        🏠 <span>HousePredict</span> AI
    </div>
</div>
""")


# =========================================================
# NAVIGATION
# =========================================================

page = st.radio(
    "",
    [
        "🏠 Home",
        "💰 Prediction",
        "🤖 AI Assistant",
        "🧠 Model",
        "📊 Project"
    ],
    horizontal=True
)

st.divider()


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    col1, col2 = st.columns([2.2, 1])

    with col1:

        st.html("""
        <div class="hero">

            <div class="hero-tag">
                🤖 Machine Learning + Real Estate
            </div>

            <div class="hero-title">
                Smart <span>House Price</span><br>
                Prediction
            </div>

            <div class="hero-text">
                Estimate house prices using a Gradient Boosting
                machine learning model trained on important
                house-related features.
            </div>

        </div>
        """)

    with col2:

        st.html("""
        <div class="hero">
            <div style="
                font-size:100px;
                text-align:center;
                padding:30px;
            ">
                🏠
            </div>
        </div>
        """)


    st.html("""
    <div class="section-heading">
        ⚡ Project Overview
    </div>

    <div class="section-subtitle">
        A machine learning based house price prediction system.
    </div>
    """)


    c1, c2, c3 = st.columns(3)

    with c1:

        st.html("""
        <div class="info-card">
            <div class="info-icon">🚀</div>
            <div class="info-title">Algorithm</div>
            <div class="info-text">
                Gradient Boosting Regressor
            </div>
        </div>
        """)

    with c2:

        st.html("""
        <div class="info-card">
            <div class="info-icon">🎯</div>
            <div class="info-title">Problem Type</div>
            <div class="info-text">
                Regression
            </div>
        </div>
        """)

    with c3:

        st.html("""
        <div class="info-card">
            <div class="info-icon">💰</div>
            <div class="info-title">Output</div>
            <div class="info-text">
                House Sale Price
            </div>
        </div>
        """)


# =========================================================
# PREDICTION
# =========================================================

elif page == "💰 Prediction":

    st.html("""
    <div class="section-heading">
        💰 House Price Prediction
    </div>

    <div class="section-subtitle">
        Enter house information to estimate its selling price.
    </div>

    <div class="card">
        <div class="card-title">
            🏠 House Information
        </div>

        <div class="card-text">
            Enter the required features used by the
            Gradient Boosting model.
        </div>
    </div>
    """)


    # ROW 1

    col1, col2, col3 = st.columns(3)

    with col1:

        overall_qual = st.number_input(
            "Overall Quality",
            min_value=1,
            max_value=10,
            value=5
        )

    with col2:

        gr_liv_area = st.number_input(
            "Living Area (sq ft)",
            min_value=100,
            max_value=10000,
            value=1500
        )

    with col3:

        garage_cars = st.number_input(
            "Garage Capacity",
            min_value=0,
            max_value=5,
            value=2
        )


    # ROW 2

    col4, col5, col6 = st.columns(3)

    with col4:

        total_bsmt_sf = st.number_input(
            "Basement Area (sq ft)",
            min_value=0,
            max_value=5000,
            value=1000
        )

    with col5:

        full_bath = st.number_input(
            "Full Bathrooms",
            min_value=0,
            max_value=5,
            value=2
        )

    with col6:

        year_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=2026,
            value=2000
        )


    # ROW 3

    col7, col8, col9 = st.columns(3)

    with col7:

        bedroom_abvgr = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=10,
            value=3
        )


    # PREDICT BUTTON

    if st.button("🔮 Predict House Price"):

        input_data = pd.DataFrame({

            "OverallQual": [overall_qual],

            "GrLivArea": [gr_liv_area],

            "GarageCars": [garage_cars],

            "TotalBsmtSF": [total_bsmt_sf],

            "FullBath": [full_bath],

            "YearBuilt": [year_built],

            "BedroomAbvGr": [bedroom_abvgr]

        })


        try:

            prediction = model.predict(input_data)[0]

            st.html(f"""
            <div class="prediction-card">

                <div style="font-size:35px;">
                    🏠
                </div>

                <div class="prediction-title">
                    Estimated House Sale Price
                </div>

                <div class="prediction-value">
                    ₹{prediction:,.2f}
                </div>

                <div style="
                    color:#d0c4d8;
                    font-size:13px;
                ">
                    Predicted using Gradient Boosting Regressor
                </div>

            </div>
            """)

        except Exception as e:

            st.error(
                f"❌ Prediction failed: {e}"
            )


# =========================================================
# AI ASSISTANT
# =========================================================

elif page == "🤖 AI Assistant":

    st.html("""
    <div class="section-heading">
        🤖 HousePredict AI Assistant
    </div>

    <div class="section-subtitle">
        Ask questions about the dataset, Gradient Boosting
        and this House Price Prediction project.
    </div>
    """)


    st.html("""
    <div class="card">

        <div class="card-title">
            💬 Project AI Assistant
        </div>

        <div class="card-text">
            I can answer questions about the dataset,
            features, regression, Gradient Boosting,
            model evaluation and this project.
        </div>

    </div>
    """)


    question = st.chat_input(
        "Ask something about the project..."
    )


    if question:

        st.chat_message("user").write(question)

        q = question.lower()


        if "objective" in q or "aim" in q:

            answer = """
            The main objective of this project is to develop
            a machine learning regression model that predicts
            house sale prices based on important house-related
            features.
            """


        elif "gradient boosting" in q:

            answer = """
            Gradient Boosting Regressor is an ensemble machine
            learning algorithm that combines multiple decision
            trees sequentially.

            Each new tree focuses on correcting errors made by
            previous trees.

            In this project, Gradient Boosting was selected as
            the best regression model.
            """


        elif "feature" in q or "features" in q:

            answer = """
            The features used in this project are:

            • Overall Quality
            • Living Area
            • Garage Capacity
            • Basement Area
            • Full Bathrooms
            • Year Built
            • Bedrooms

            The target variable is SalePrice.
            """


        elif "target" in q:

            answer = """
            The target variable is 'SalePrice'.

            It represents the selling price of the house that
            the model is trained to predict.
            """


        elif "regression" in q:

            answer = """
            Regression is a supervised machine learning
            technique used to predict continuous numerical values.

            In this project, regression is used to predict
            house sale prices.
            """


        elif "dataset" in q:

            answer = """
            This project uses the House Price dataset.

            It contains different house-related features and
            the target variable SalePrice.
            """


        elif "accuracy" in q:

            answer = """
            Accuracy is mainly used for classification problems.

            Since this project is a regression problem, metrics
            such as R² Score, MAE and RMSE are used for evaluation.
            """


        elif "mae" in q:

            answer = """
            MAE stands for Mean Absolute Error.

            It measures the average absolute difference between
            actual and predicted house prices.

            Lower MAE generally indicates better performance.
            """


        elif "rmse" in q:

            answer = """
            RMSE stands for Root Mean Squared Error.

            It measures prediction error and gives greater
            importance to larger errors.

            Lower RMSE generally indicates better performance.
            """


        elif "r2" in q or "r²" in q:

            answer = """
            R² Score represents how well the model explains
            the variation in the target variable.

            A value closer to 1 generally indicates better
            regression performance.
            """


        elif "technology" in q or "technologies" in q:

            answer = """
            Technologies used in this project include:

            • Python
            • Pandas
            • NumPy
            • Scikit-learn
            • Joblib
            • Streamlit
            • Machine Learning
            """


        elif "future" in q:

            answer = """
            Future scope includes:

            • Improving prediction accuracy
            • Hyperparameter tuning
            • Trying advanced regression algorithms
            • Adding more useful house features
            • Improving the user interface
            • Real-world deployment
            """


        else:

            answer = """
            I can answer questions about:

            • Project Objective
            • Dataset
            • Features
            • Target Variable
            • Regression
            • Gradient Boosting
            • MAE
            • RMSE
            • R² Score
            • Technologies
            • Future Scope

            Try asking one of these topics.
            """


        st.chat_message("assistant").write(answer)


# =========================================================
# MODEL
# =========================================================

elif page == "🧠 Model":

    st.html("""
    <div class="section-heading">
        🧠 Model Information
    </div>

    <div class="section-subtitle">
        Machine learning model used in this application.
    </div>
    """)


    c1, c2, c3 = st.columns(3)


    with c1:

        st.html("""
        <div class="info-card">
            <div class="info-icon">🚀</div>
            <div class="info-title">Algorithm</div>
            <div class="info-text">
                Gradient Boosting Regressor
            </div>
        </div>
        """)


    with c2:

        st.html("""
        <div class="info-card">
            <div class="info-icon">🎯</div>
            <div class="info-title">Model Type</div>
            <div class="info-text">
                Regression
            </div>
        </div>
        """)


    with c3:

        st.html("""
        <div class="info-card">
            <div class="info-icon">💰</div>
            <div class="info-title">Target</div>
            <div class="info-text">
                SalePrice
            </div>
        </div>
        """)


    st.html("""
    <div class="card">

        <div class="card-title">
            📌 Why Gradient Boosting?
        </div>

        <div class="card-text">
            Gradient Boosting was selected as the best model
            after comparing multiple regression algorithms.

            It builds decision trees sequentially and focuses
            on correcting errors made by previous trees.

            This helps the model capture complex relationships
            between house features and house prices.
        </div>

    </div>
    """)


    st.html("""
    <div class="card">

        <div class="card-title">
            📊 Model Evaluation
        </div>

        <div class="card-text">

            The regression model can be evaluated using:

            <br><br>

            <b>MAE</b> – Mean Absolute Error

            <br><br>

            <b>RMSE</b> – Root Mean Squared Error

            <br><br>

            <b>R² Score</b> – Coefficient of Determination

            <br><br>

            Lower MAE and RMSE and higher R² generally
            indicate better model performance.

        </div>

    </div>
    """)


# =========================================================
# PROJECT
# =========================================================

elif page == "📊 Project":

    st.html("""
    <div class="section-heading">
        📊 Project Information
    </div>

    <div class="section-subtitle">
        Complete overview of the House Price Prediction project.
    </div>
    """)


    st.html("""
    <div class="card">

        <div class="card-title">
            📌 Problem Statement
        </div>

        <div class="card-text">
            House prices depend on several factors such as
            overall quality, living area, garage capacity,
            basement area, bathrooms, year built and number
            of bedrooms.

            The objective is to develop a machine learning
            model that can estimate house prices based on
            these factors.
        </div>

    </div>
    """)


    st.html("""
    <div class="card">

        <div class="card-title">
            🎯 Project Objective
        </div>

        <div class="card-text">
            The main objective is to build a regression model
            capable of predicting house sale prices from
            house-related information.

            Multiple regression algorithms were compared and
            Gradient Boosting was selected as the best model.
        </div>

    </div>
    """)


    st.html("""
    <div class="card">

        <div class="card-title">
            🧬 Features
        </div>

        <div class="card-text">
            • Overall Quality<br>
            • Living Area<br>
            • Garage Capacity<br>
            • Basement Area<br>
            • Full Bathrooms<br>
            • Year Built<br>
            • Bedrooms
        </div>

    </div>
    """)


    st.html("""
    <div class="card">

        <div class="card-title">
            🎯 Target Variable
        </div>

        <div class="card-text">

            <b>SalePrice</b>

            <br><br>

            The target variable represents the selling price
            of the house that the model predicts.

        </div>

    </div>
    """)


    st.html("""
    <div class="card">

        <div class="card-title">
            🚀 Future Scope
        </div>

        <div class="card-text">

            • Improve model accuracy<br>
            • Hyperparameter tuning<br>
            • Try advanced regression algorithms<br>
            • Add more house-related features<br>
            • Improve user interface<br>
            • Real-world deployment

        </div>

    </div>
    """)


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">

    ✨ Machine Learning Project |
    Gradient Boosting Regressor |
    House Price Prediction

    <br><br>

    🏠 Built with Python + Streamlit + Machine Learning

</div>
""")
