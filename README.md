# Car_Price_Prediction
# 🚗 Car Price Prediction

An end-to-end machine learning project that predicts the **price (in lac PKR)** of used cars in Pakistan, deployed as an interactive **Streamlit web app**.

## 📌 Project Overview
Used car prices depend on many factors: brand, age, mileage, engine size, transmission and city. This project builds a regression model that learns these patterns from real listings data and gives an instant price estimate.

## 🔍 Workflow
- **Data Cleaning:** removed duplicates, replaced fake mileage values (e.g. `123456`) with NaN and imputed with the median, and handled price/mileage outliers using the **IQR method**.
- **Feature Engineering:** extracted `Brand`, `Model_Name` and `Variant` from the car name, and created `Car_Age`, `IsAutomatic` and `Location_Tier` (big city vs others).
- **EDA:** price distribution, age vs price, mileage vs price, brand-wise comparison, transmission impact and a correlation heatmap.
- **Preprocessing:** `ColumnTransformer` with **StandardScaler** + **OneHotEncoder**, wrapped in a `Pipeline` to avoid data leakage.

## 🤖 Models Compared
**Ridge Regression, Random Forest, XGBoost and CatBoost**, each tuned with **GridSearchCV (5-fold CV)**.

| Metric | CatBoost (Best) |
|---|---|
| **MAE** | 4.58 lac |
| **RMSE** | 6.59 lac |
| **R² Score** | 0.863 |
| **MAPE** | 28.3% |

🏆 **CatBoost** performed best and is used in the final app.

## 🌐 Streamlit App
Select a car, city, year, mileage, engine capacity and transmission to get an **estimated price** with an expected error range.

## 🛠️ Tech Stack
`Python` • `Pandas` • `NumPy` • `Matplotlib` • `Seaborn` • `Scikit-learn` • `XGBoost` • `CatBoost` • `Streamlit`

## ⚠️ Limitations
Extreme-price outliers were capped during cleaning, so estimates for very expensive luxury cars may be on the lower side.
