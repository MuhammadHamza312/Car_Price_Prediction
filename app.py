"""
app.py - Modern Streamlit Web Application for Pakistani Car Price Prediction
Powered by CatBoost Regressor
"""

import os
import pickle
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Pakistani Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.15), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        max-width: 800px;
        line-height: 1.5;
        font-weight: 400;
    }
    
    /* Card Containers */
    .form-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
    }
    
    /* Prediction Result Card */
    .result-card {
        background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%);
        border: 2px solid #0d9488;
        border-radius: 20px;
        padding: 2.2rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 20px 25px -5px rgba(13, 148, 136, 0.12);
        animation: fadeIn 0.4s ease-in-out;
    }
    .result-header {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #0f766e;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    .result-price-lac {
        font-size: 3.2rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .result-price-pkr {
        font-size: 1.35rem;
        font-weight: 600;
        color: #0d9488;
        margin-top: 0.4rem;
        margin-bottom: 1.2rem;
    }
    .result-range {
        display: inline-block;
        background-color: #ccfbf1;
        color: #115e59;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.5rem 1.2rem;
        border-radius: 9999px;
        border: 1px solid #99f6e4;
    }
    
    /* Warning Box */
    .cap-warning {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-top: 1.2rem;
        color: #92400e;
        font-size: 0.92rem;
        text-align: left;
    }

    /* Metric Badges */
    .metric-badge {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .metric-badge-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-badge-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    /* Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
        color: white;
        font-weight: 700;
        font-size: 1.15rem;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(13, 148, 136, 0.39);
        transition: all 0.2s ease-in-out;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(13, 148, 136, 0.45);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained CatBoost model pipeline...")
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'car_price_catboost_pipeline.pkl')
    meta_path = os.path.join(base_dir, 'app_metadata.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_model.py first.")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}. Please run train_model.py first.")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)

    return model, metadata


def format_pkr(amount_in_lac):
    pkr_amount = int(round(amount_in_lac * 100000))
    return f"PKR {pkr_amount:,}"


def main():
    # Load model and metadata
    try:
        model, metadata = load_artifacts()
    except Exception as e:
        st.error(f"Error loading model resources: {str(e)}")
        st.info("Make sure you have generated the model files by executing `python train_model.py`.")
        st.stop()
        return

    feature_columns = metadata['feature_columns']
    car_names = metadata['car_names']
    car_lookup = metadata['car_lookup']
    engine_by_car = metadata['engine_by_car']
    locations = metadata['locations']
    fuel_options = metadata.get('fuel_options', [1])
    fuel_default = metadata.get('fuel_default', 1)
    year_min = metadata.get('year_min', 1980)
    year_max = metadata.get('year_max', 2026)
    mileage_min = metadata.get('mileage_min', 1)
    mileage_max = metadata.get('mileage_max', 500000)
    engine_min = metadata.get('engine_min', 600)
    engine_max = metadata.get('engine_max', 6000)
    price_max = metadata.get('price_max', 100.0)
    metrics = metadata.get('metrics', {'MAE': 4.58, 'RMSE': 6.59, 'R2': 0.863, 'MAPE': 28.3})

    # Sidebar: About & Model Information
    with st.sidebar:
        st.markdown("### 🚘 Pakistani Car Price AI")
        st.write(
            "An end-to-end Machine Learning web application designed to provide fair valuation for "
            "used cars in Pakistan's automotive market."
        )

        st.markdown("---")
        st.markdown("### 📊 Best Model: CatBoost")
        st.caption("Selected after rigorous benchmark against Ridge, Random Forest, and XGBoost.")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Test MAE", f"{metrics.get('MAE', 4.58):.2f} Lac")
            st.metric("R² Score", f"{metrics.get('R2', 0.863):.3f}")
        with col_m2:
            st.metric("RMSE", f"{metrics.get('RMSE', 6.59):.2f}")
            st.metric("MAPE", f"{metrics.get('MAPE', 28.3):.1f}%")

        st.markdown("---")
        st.markdown("### ℹ️ Pricing Context")
        st.markdown(
            "- **1 Lac PKR** = Rs 100,000\n"
            "- Target is trained on Pakistani listings\n"
            "- Outliers clipped during training for maximum robustness on mainstream vehicles."
        )
        st.markdown("---")
        st.caption("Built with Streamlit & CatBoost")

    # Main Hero Header
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🚗 Pakistani Car Price Predictor</div>
        <div class="hero-subtitle">
            Instant, data-driven market valuations for used Pakistani automobiles.
            Select your vehicle specifications below to estimate its realistic fair price.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Input Form Layout
    st.markdown("#### ⚙️ Vehicle Specifications")
    
    col1, col2 = st.columns(2, gap="large")

    with col1:
        # Searchable car selection
        default_car_idx = 0
        preferred_defaults = ["Toyota Corolla Altis 1 1.6", "Honda Civic Oriel 1.6", "Suzuki Alto R AGS", "Suzuki Mehran II"]
        for pref in preferred_defaults:
            if pref in car_names:
                default_car_idx = car_names.index(pref)
                break

        selected_car = st.selectbox(
            "Car Model & Variant *",
            options=car_names,
            index=default_car_idx,
            help="Select the exact car brand and model name"
        )

        # Retrieve brand, model_name, and variant for info
        car_info = car_lookup.get(selected_car, {'Brand': '', 'Model_Name': '', 'Variant': ''})

        # Location
        default_loc_idx = 0
        for city in ["Lahore", "Karachi", "Islamabad"]:
            if city in locations:
                default_loc_idx = locations.index(city)
                break

        selected_location = st.selectbox(
            "Registered City / Location *",
            options=locations,
            index=default_loc_idx,
            help="Major metropolitan tiers receive localized pricing adjustments"
        )

        # Transmission
        transmission = st.radio(
            "Transmission Type *",
            options=["Automatic", "Manual"],
            index=0,
            horizontal=True,
            help="Choose vehicle transmission"
        )
        is_automatic = 1 if transmission == "Automatic" else 0

    with col2:
        # Model Year
        current_year = 2026
        default_year = min(max(2018, year_min), year_max)
        selected_year = st.slider(
            "Model Manufacturing Year *",
            min_value=int(year_min),
            max_value=int(year_max),
            value=int(default_year),
            step=1,
            help=f"Select between {year_min} and {year_max}"
        )

        # Mileage
        selected_mileage = st.number_input(
            "Total Mileage (in Kilometers) *",
            min_value=int(mileage_min),
            max_value=int(mileage_max),
            value=int(min(max(60000, mileage_min), mileage_max)),
            step=5000,
            help="Enter total kilometers driven"
        )

        # Engine Capacity default from engine_by_car
        default_engine = engine_by_car.get(selected_car, 1300)
        selected_engine = st.number_input(
            "Engine Capacity (in CC) *",
            min_value=int(engine_min),
            max_value=int(engine_max),
            value=int(default_engine),
            step=100,
            help=f"Defaulted to median for {selected_car}"
        )

        # Fuel selection only if more than 1 distinct option
        if len(fuel_options) > 1:
            selected_fuel = st.selectbox(
                "Fuel Type",
                options=fuel_options,
                index=fuel_options.index(fuel_default) if fuel_default in fuel_options else 0
            )
        else:
            selected_fuel = fuel_default

    # Predict Button
    st.write("")
    predict_clicked = st.button("🔍 Predict Estimated Car Price", use_container_width=True)

    if predict_clicked:
        # Prepare inputs exactly as expected by pipeline
        brand = car_info['Brand']
        model_name = car_info['Model_Name']
        variant = car_info['Variant']
        car_age = 2026 - selected_year
        big_cities = ['Karachi', 'Lahore', 'Islamabad']
        location_tier = 1 if selected_location.strip().title() in big_cities else 0

        input_row = {
            'Car Name': selected_car,
            'Location': selected_location,
            'Year': selected_year,
            'Mileage (km)': selected_mileage,
            'Fuel (Diesel/Petrol/Hybrid)': selected_fuel,
            'Engine Capacity': selected_engine,
            'IsAutomatic': is_automatic,
            'Brand': brand,
            'Model_Name': model_name,
            'Variant': variant,
            'Car_Age': car_age,
            'Location_Tier': location_tier,
        }

        try:
            # Construct DataFrame in exact feature_columns ordering
            input_df = pd.DataFrame([input_row])[feature_columns]

            # Model prediction
            raw_pred = float(model.predict(input_df)[0])
            pred_price = max(raw_pred, 0.5)  # Sanity floor

            mae_val = metrics.get('MAE', 4.58)
            low_bound = max(pred_price - mae_val, 0.2)
            high_bound = pred_price + mae_val

            # Display Result Card
            st.markdown(f"""
            <div class="result-card">
                <div class="result-header">Estimated Market Valuation</div>
                <div class="result-price-lac">{pred_price:.2f} Lac PKR</div>
                <div class="result-price-pkr">{format_pkr(pred_price)}</div>
                <div class="result-range">
                    Realistic Range: {low_bound:.2f} Lac – {high_bound:.2f} Lac PKR (±{mae_val:.2f} Lac MAE)
                </div>
            </div>
            """, unsafe_allow_html=True)

            # High price cap warning
            if pred_price >= (price_max * 0.90):
                st.markdown(f"""
                <div class="cap-warning">
                    ⚠️ <strong>Notice:</strong> This predicted price ({pred_price:.2f} Lac) is near or above the dataset's 
                    training cap ({price_max:.1f} Lac PKR). To ensure high generalization across mainstream vehicles, 
                    extreme luxury price outliers were capped in data preprocessing.
                </div>
                """, unsafe_allow_html=True)

            # Details Expander
            with st.expander("📋 View Analyzed Feature Vector Details", expanded=False):
                col_det1, col_det2, col_det3 = st.columns(3)
                with col_det1:
                    st.write(f"**Brand:** {brand}")
                    st.write(f"**Model Name:** {model_name}")
                    st.write(f"**Variant:** {variant if variant else 'Standard'}")
                with col_det2:
                    st.write(f"**Vehicle Age:** {car_age} years")
                    st.write(f"**Engine:** {selected_engine} CC")
                    st.write(f"**Transmission:** {transmission}")
                with col_det3:
                    st.write(f"**Location:** {selected_location}")
                    st.write(f"**Location Tier:** Tier 1 (Metro)" if location_tier == 1 else "**Location Tier:** Tier 2 / Regional")
                    st.write(f"**Mileage:** {selected_mileage:,} km")

        except Exception as err:
            st.error(f"Error occurred during price calculation: {str(err)}")
            st.info("Please verify your input values and try again.")


if __name__ == '__main__':
    main()
