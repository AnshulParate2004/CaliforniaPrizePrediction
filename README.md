# California Housing Price Prediction - Streamlit App
A beautiful and user-friendly web application for predicting California housing prices using machine learning models.

## Features

- 🏠 Interactive user interface
- 🤖 Three prediction models:
  - LightGBM Regressor
  - XGBoost Regressor
  - Ensemble (Average of both)
- 📊 Real-time predictions
- 🎨 Clean and modern design
- 📱 Responsive layout

## Project Structure

```
CaliforniaPrizePrediction/
├── backend/
│   ├── main.py              # Streamlit app
│   ├── app.py               # FastAPI app (original)
│   └── requirements.txt     # Python dependencies
└── models/
    ├── tuned_lightgbm_regressor_model.pkl
    └── tuned_xgboost_regressor_model.pkl
```

## Usage

1. Enter house features in the input fields
2. Select ocean proximity from the dropdown
3. Choose your preferred model (LightGBM, XGBoost, or Ensemble)
4. Click "Predict House Price"
5. View the predicted median house value

## Input Features

- **Longitude**: Longitude coordinate of the house
- **Latitude**: Latitude coordinate of the house
- **Housing Median Age**: Median age of houses in the block (years)
- **Total Rooms**: Total number of rooms in the block
- **Total Bedrooms**: Total number of bedrooms in the block
- **Population**: Total population in the block
- **Households**: Number of households in the block
- **Median Income**: Median income in tens of thousands ($10,000s)
- **Ocean Proximity**: Distance/relation to ocean (INLAND, <1H OCEAN, NEAR OCEAN, NEAR BAY, ISLAND)
- **Bedroom Ratio**: Ratio of bedrooms to total rooms
- **Household Rooms**: Average rooms per household

## Notes

- Model files are loaded from `...\CaliforniaPrizePrediction\models`
- For deployment, ensure model files are accessible (commit to repo if under 100MB or use cloud storage)
- The app uses caching to load models efficiently

## License
MIT License
