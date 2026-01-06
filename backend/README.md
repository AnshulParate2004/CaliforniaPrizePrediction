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

## Installation

1. **Install dependencies:**
```bash
cd D:\Learning\CaliforniaPrizePrediction\backend
pip install -r requirements.txt
```

## Running Locally

Run the Streamlit app:
```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## Deployment Options

### 1. Streamlit Community Cloud (Recommended - Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository, branch, and main file (`main.py`)
6. Click "Deploy"

**Important:** For Streamlit Cloud, you'll need to upload your model files to the repository or use a cloud storage service like Google Drive or AWS S3.

### 2. Heroku

1. Install Heroku CLI
2. Create a `Procfile`:
```
web: streamlit run main.py --server.port=$PORT
```

3. Create a `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

4. Deploy:
```bash
heroku login
heroku create your-app-name
git push heroku main
```

### 3. Render (Free Tier Available)

1. Go to [render.com](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repository
4. Set the following:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run main.py --server.port=$PORT --server.address=0.0.0.0`

### 4. AWS EC2

1. Launch an EC2 instance
2. SSH into the instance
3. Install Python and dependencies
4. Run the app with:
```bash
streamlit run main.py --server.port=8501 --server.address=0.0.0.0
```

### 5. Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t california-housing-predictor .
docker run -p 8501:8501 california-housing-predictor
```

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

- Model files are loaded from `D:\Learning\CaliforniaPrizePrediction\models`
- For deployment, ensure model files are accessible (commit to repo if under 100MB or use cloud storage)
- The app uses caching to load models efficiently

## Troubleshooting

**Models not loading:**
- Check that model paths are correct
- Ensure models directory exists
- Verify model files are not corrupted

**Port already in use:**
```bash
streamlit run main.py --server.port=8502
```

**Dependencies issues:**
```bash
pip install --upgrade -r requirements.txt
```

## License

MIT License
