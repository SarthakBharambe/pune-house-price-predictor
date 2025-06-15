
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap



st.set_page_config(page_title="🏠 Pune House Price Predictor", layout="centered")

def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('https://images.unsplash.com/photo-1748585462833-68f59d09c1d4?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }}
        input[type="range"] {{
            accent-color: #00BFFF;
        }}
        .main-container {{
            background: rgba(255, 255, 255, 0.75);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            max-width: 700px;
            margin: auto;
            margin-top: 30px;
        }}
        .stButton button {{
            background-color: #008080;
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-weight: 600;
        }}
        html, body, [class*="css"]  {{
            font-family: 'Segoe UI', sans-serif;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

# Load model
model = pickle.load(open("pune_price_model.pkl", "rb"))
explainer = shap.Explainer(model)


@st.cache_data
def load_data():
    return pd.read_csv("Pune_house_data.csv")

df = load_data()

# Encode categorical variables
area_type_map = {t: i for i, t in enumerate(sorted(df['area_type'].dropna().unique()))}
all_locations = sorted(df['site_location'].dropna().unique())
site_location_map = {t: i for i, t in enumerate(all_locations)}

# UI
st.title("🏠 Pune House Price Prediction App")
tab1, tab2 = st.tabs(["🏠 Predictor", "📊 Data Insights"])

with tab1:

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.subheader("Enter Property Details")

    col1, col2 = st.columns(2)
    with col1:
        total_sqft = st.slider("Total Square Feet", 300, 10000, 1000, step=50)
        bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
        bath = st.selectbox("Bathrooms", [1, 2, 3, 4, 5])
    with col2:
        balcony = st.selectbox("Balconies", [0, 1, 2, 3])
        area_type = st.selectbox("Area Type", list(area_type_map.keys()))
        site_location = st.selectbox("Location", all_locations)

    encoded_area_type = area_type_map.get(area_type, -1)
    encoded_location = site_location_map.get(site_location, -1)
    if encoded_area_type == -1 or encoded_location == -1:
        st.error("Invalid location or area type selected.")
    else:
        input_features = np.array([[total_sqft, bhk, bath, balcony, encoded_area_type, encoded_location]])

        try:
            predicted_price = model.predict(input_features)[0]
            st.markdown(f"<h4 style='color: white;'>✅ Predicted Price: ₹ {round(predicted_price, 2)} Lakhs</h4>", unsafe_allow_html=True)

            # SHAP explanation
            shap_values = explainer(input_features)
            st.markdown("#### 🧠 Feature Impact on Price (SHAP)")
            import matplotlib.pyplot as plt
            fig = plt.figure()
            shap.plots.bar(shap_values[0], show=False)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


    st.markdown('</div>', unsafe_allow_html=True)



with tab2:
    st.markdown("### Sample Data")
    st.dataframe(df.head())

    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.markdown("### BHK vs Price Distribution")
    df_clean = df.copy()
    df_clean['bhk'] = df_clean['size'].apply(lambda x: int(str(x).split(' ')[0]) if isinstance(x, str) and str(x)[0].isdigit() else np.nan)
    df_clean = df_clean.dropna(subset=['bhk', 'price'])

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='bhk', y='price', data=df_clean, ax=ax2)
    ax2.set_title("Price Distribution by BHK Count")
    ax2.set_xlabel("BHK")
    ax2.set_ylabel("Price (Lakhs)")
    st.pyplot(fig2)
    st.markdown("### 📍 Average Price per Location")

# Compute average price per location
avg_price_by_location = df.groupby("site_location")["price"].mean().sort_values(ascending=False).head(20)

fig3, ax3 = plt.subplots(figsize=(12, 6))
sns.barplot(x=avg_price_by_location.values, y=avg_price_by_location.index, palette="viridis", ax=ax3)
ax3.set_xlabel("Average Price (Lakhs)")
ax3.set_ylabel("Location")
ax3.set_title("Top 20 Locations by Average Price")
st.pyplot(fig3)
