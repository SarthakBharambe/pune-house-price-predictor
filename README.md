# 🏠 Pune House Price Predictor

A modern, visually rich web app built using **Streamlit** that predicts house prices in Pune based on key property details. Powered by **XGBoost**, enriched with **SHAP explainability**, and styled with a custom UI for a professional look.

---

## 🚀 Live Demo
👉 [View the App](https://pune-house-price-predictor-jb8zbultqbhmkgoeqnazfx.streamlit.app)

---

## 🧠 Features

- 🎯 Predicts house price using:
  - Square feet
  - Number of bedrooms (BHK)
  - Bathrooms
  - Balconies
  - Area type
  - Site location (full list)

- 🎨 **Modern UI**
  - Background image
  - Dark mode theme
  - Styled sliders, dropdowns, and buttons

- 📊 **Data Insights** tab:
  - Correlation heatmap
  - BHK vs Price distribution
  - Sample dataset preview

- 💬 **Model Explainability** using SHAP
  - See how each feature influences the predicted price

---

## 📁 Files
| File | Purpose |
|------|---------|
| `streamlit_app_final_all_locations.py` | ✅ Final app file with all features |
| `pune_price_model.pkl` | Trained XGBoost model |
| `Pune_house_data.csv` | Cleaned dataset |
| `requirements.txt` | All dependencies for deployment |

---

## 🛠️ Tech Stack
- Python 3.9+
- Streamlit
- Pandas, NumPy
- XGBoost
- SHAP
- Seaborn, Matplotlib
- Scikit-learn

---

## 🧪 How to Run Locally
```bash
# 1. Clone the repo
https://github.com/SarthakBharambe07/pune-house-price-predictor.git
cd pune-house-price-predictor

# 2. Create environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run streamlit_app_final_all_locations.py
```

---

## 📌 Author
**Sarthak Bharambe**  
Connect on [LinkedIn](https://www.linkedin.com/in/sarthak-bharambe/) | GitHub: [@SarthakBharambe07](https://github.com/SarthakBharambe07)

---

## 📄 License
This project is for academic and portfolio purposes.

 
