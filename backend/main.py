import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        models_path = Path(r"...\CaliforniaPrizePrediction\models")
        lightgbm_model = joblib.load(models_path / "tuned_lightgbm_regressor_model.pkl")
        xgboost_model = joblib.load(models_path / "tuned_xgboost_regressor_model.pkl")
        return lightgbm_model, xgboost_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

lightgbm_model, xgboost_model = load_models()

# Header
st.markdown('<h1 class="main-header">🏠 California Housing Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for model selection
st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.radio(
    "Choose Prediction Model:",
    ["LightGBM", "XGBoost", "Ensemble (Average)"],
    help="Select which model to use for prediction"
)

st.sidebar.markdown("---")
st.sidebar.header("📊 About")
st.sidebar.info(
    """
    This app predicts California housing prices using machine learning models.
    
    **Models Used:**
    - LightGBM Regressor
    - XGBoost Regressor
    - Ensemble (Average of both)
    
    **Data Source:**
    California Housing Dataset
    """
)

# Main content - Input form
st.header("📝 Enter House Features")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Location & Demographics")
    longitude = st.number_input(
        "Longitude",
        value=-122.23,
        format="%.4f",
        help="Longitude coordinate of the house"
    )
    
    latitude = st.number_input(
        "Latitude",
        value=37.88,
        format="%.4f",
        help="Latitude coordinate of the house"
    )
    
    housing_median_age = st.slider(
        "Housing Median Age (years)",
        min_value=1,
        max_value=100,
        value=41,
        help="Median age of houses in the block"
    )
    
    median_income = st.number_input(
        "Median Income (in $10,000s)",
        min_value=0.01,
        value=8.3252,
        format="%.4f",
        help="Median income of households in tens of thousands"
    )
    
    population = st.number_input(
        "Population",
        min_value=1,
        value=322,
        step=1,
        help="Total population in the block"
    )

with col2:
    st.subheader("House Characteristics")
    total_rooms = st.number_input(
        "Total Rooms",
        min_value=1,
        value=880,
        step=1,
        help="Total number of rooms in the block"
    )
    
    total_bedrooms = st.number_input(
        "Total Bedrooms",
        min_value=1,
        value=129,
        step=1,
        help="Total number of bedrooms in the block"
    )
    
    households = st.number_input(
        "Households",
        min_value=1,
        value=126,
        step=1,
        help="Number of households in the block"
    )
    
    bedroom_ratio = st.number_input(
        "Bedroom Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.1466,
        format="%.4f",
        help="Ratio of bedrooms to total rooms"
    )
    
    household_rooms = st.number_input(
        "Household Rooms",
        min_value=0.0,
        value=6.9841,
        format="%.4f",
        help="Average rooms per household"
    )

# Ocean Proximity Selection
st.subheader("🌊 Ocean Proximity")
ocean_proximity = st.selectbox(
    "Select Ocean Proximity:",
    ["INLAND", "<1H OCEAN", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
)

# Set ocean proximity one-hot encoding
ocean_proximity_1H_OCEAN = 1 if ocean_proximity == "<1H OCEAN" else 0
ocean_proximity_INLAND = 1 if ocean_proximity == "INLAND" else 0
ocean_proximity_ISLAND = 1 if ocean_proximity == "ISLAND" else 0
ocean_proximity_NEAR_BAY = 1 if ocean_proximity == "NEAR BAY" else 0
ocean_proximity_NEAR_OCEAN = 1 if ocean_proximity == "NEAR OCEAN" else 0

st.markdown("---")

# Prediction function
def make_prediction(model, features_dict, model_name="Model"):
    try:
        # Create DataFrame with proper column names
        input_df = pd.DataFrame([features_dict])
        
        # Try direct prediction first
        try:
            prediction = model.predict(input_df)[0]
            return round(float(prediction), 2)
        except (AttributeError, TypeError):
            # If scikit-learn wrapper fails, try using the booster directly
            if hasattr(model, 'booster_'):
                # LightGBM booster - use raw booster predict
                prediction = model.booster_.predict(input_df.values)[0]
                return round(float(prediction), 2)
            elif hasattr(model, '_Booster'):
                # XGBoost booster
                import xgboost as xgb
                dmatrix = xgb.DMatrix(input_df)
                prediction = model._Booster.predict(dmatrix)[0]
                return round(float(prediction), 2)
            else:
                raise
                
    except Exception as e:
        st.error(f"❌ {model_name} prediction error: {str(e)}")
        raise

# Predict button
if st.button("🔮 Predict House Price", type="primary"):
    if lightgbm_model is None or xgboost_model is None:
        st.error("Models not loaded properly. Please check the model files.")
    else:
        # Prepare features for both models (they have different naming conventions)
        features_lightgbm = {
            'longitude': longitude,
            'latitude': latitude,
            'housing_median_age': housing_median_age,
            'total_rooms': total_rooms,
            'total_bedrooms': total_bedrooms,
            'population': population,
            'households': households,
            'median_income': median_income,
            'ocean_proximity_1H_OCEAN': ocean_proximity_1H_OCEAN,  # LightGBM uses underscores
            'ocean_proximity_INLAND': ocean_proximity_INLAND,
            'ocean_proximity_ISLAND': ocean_proximity_ISLAND,
            'ocean_proximity_NEAR_BAY': ocean_proximity_NEAR_BAY,
            'ocean_proximity_NEAR_OCEAN': ocean_proximity_NEAR_OCEAN,
            'bedroom_ratio': bedroom_ratio,
            'household_rooms': household_rooms
        }
        
        features_xgboost = {
            'longitude': longitude,
            'latitude': latitude,
            'housing_median_age': housing_median_age,
            'total_rooms': total_rooms,
            'total_bedrooms': total_bedrooms,
            'population': population,
            'households': households,
            'median_income': median_income,
            'ocean_proximity_1H OCEAN': ocean_proximity_1H_OCEAN,  # XGBoost uses spaces
            'ocean_proximity_INLAND': ocean_proximity_INLAND,
            'ocean_proximity_ISLAND': ocean_proximity_ISLAND,
            'ocean_proximity_NEAR BAY': ocean_proximity_NEAR_BAY,
            'ocean_proximity_NEAR OCEAN': ocean_proximity_NEAR_OCEAN,
            'bedroom_ratio': bedroom_ratio,
            'household_rooms': household_rooms
        }
        
        # Make predictions
        with st.spinner("Calculating predictions..."):
            if model_choice == "LightGBM":
                prediction = make_prediction(lightgbm_model, features_lightgbm, "LightGBM")
                st.success("✅ Prediction Complete!")
                st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="color: #1f77b4;">LightGBM Prediction</h2>
                    <h1 style="color: #2ecc71;">${prediction:,.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
                
            elif model_choice == "XGBoost":
                prediction = make_prediction(xgboost_model, features_xgboost, "XGBoost")
                st.success("✅ Prediction Complete!")
                st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="color: #1f77b4;">XGBoost Prediction</h2>
                    <h1 style="color: #2ecc71;">${prediction:,.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
                
            else:  # Ensemble
                lgbm_pred = make_prediction(lightgbm_model, features_lightgbm, "LightGBM")
                xgb_pred = make_prediction(xgboost_model, features_xgboost, "XGBoost")
                ensemble_pred = round((lgbm_pred + xgb_pred) / 2, 2)
                
                st.success("✅ Prediction Complete!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3 style="color: #1f77b4;">LightGBM</h3>
                        <h2 style="color: #2ecc71;">${lgbm_pred:,.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3 style="color: #1f77b4;">XGBoost</h3>
                        <h2 style="color: #2ecc71;">${xgb_pred:,.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3 style="color: #e74c3c;">Ensemble Average</h3>
                        <h2 style="color: #f39c12;">${ensemble_pred:,.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)

# Display input summary
with st.expander("📋 View Input Summary"):
    input_df = pd.DataFrame({
        'Feature': [
            'Longitude', 'Latitude', 'Housing Median Age', 'Total Rooms',
            'Total Bedrooms', 'Population', 'Households', 'Median Income',
            'Ocean Proximity (<1H OCEAN)', 'Ocean Proximity (INLAND)',
            'Ocean Proximity (ISLAND)', 'Ocean Proximity (NEAR BAY)',
            'Ocean Proximity (NEAR OCEAN)', 'Bedroom Ratio', 'Household Rooms'
        ],
        'Value': [
            longitude, latitude, housing_median_age, total_rooms,
            total_bedrooms, population, households, median_income,
            ocean_proximity_1H_OCEAN, ocean_proximity_INLAND,
            ocean_proximity_ISLAND, ocean_proximity_NEAR_BAY,
            ocean_proximity_NEAR_OCEAN, bedroom_ratio, household_rooms
        ]
    })
    st.dataframe(input_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>Built with Streamlit 🎈 | Powered by LightGBM & XGBoost</p>
    </div>
    """,
    unsafe_allow_html=True
)
