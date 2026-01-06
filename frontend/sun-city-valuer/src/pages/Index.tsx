import { useState } from "react";
import { PredictionForm } from "@/components/PredictionForm";
import { PredictionResult } from "@/components/PredictionResult";
import { toast } from "sonner";
import californiaHero from "@/assets/california-hero.jpg";

interface FormData {
  longitude: string;
  latitude: string;
  housingMedianAge: string;
  totalRooms: string;
  totalBedrooms: string;
  population: string;
  households: string;
  medianIncome: string;
  oceanProximity: string;
  bedroomRatio: string;
  householdRooms: string;
}

interface EnsemblePrediction {
  lightgbm_prediction: number;
  xgboost_prediction: number;
  mean_prediction: number;
}

// Update this with your FastAPI backend URL
const API_BASE_URL = "http://localhost:8000"; // Change to your deployed API URL

const Index = () => {
  const [prediction, setPrediction] = useState<number | null>(null);
  const [ensemblePrediction, setEnsemblePrediction] = useState<EnsemblePrediction | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const transformOceanProximity = (value: string) => {
    return {
      ocean_proximity_1H_OCEAN: value === "<1H OCEAN" ? 1 : 0,
      ocean_proximity_INLAND: value === "INLAND" ? 1 : 0,
      ocean_proximity_ISLAND: value === "ISLAND" ? 1 : 0,
      ocean_proximity_NEAR_BAY: value === "NEAR BAY" ? 1 : 0,
      ocean_proximity_NEAR_OCEAN: value === "NEAR OCEAN" ? 1 : 0,
    };
  };

  const handlePredict = async (data: FormData, selectedModel: 'lightgbm' | 'xgboost' | 'ensemble') => {
    setIsLoading(true);
    setPrediction(null);
    setEnsemblePrediction(null);
    
    try {
      const oceanProximityEncoded = transformOceanProximity(data.oceanProximity);
      
      const requestBody = {
        longitude: parseFloat(data.longitude),
        latitude: parseFloat(data.latitude),
        housing_median_age: parseFloat(data.housingMedianAge),
        total_rooms: parseFloat(data.totalRooms),
        total_bedrooms: parseFloat(data.totalBedrooms),
        population: parseFloat(data.population),
        households: parseFloat(data.households),
        median_income: parseFloat(data.medianIncome),
        ...oceanProximityEncoded,
        bedroom_ratio: parseFloat(data.bedroomRatio),
        household_rooms: parseFloat(data.householdRooms),
      };

      const endpoint = selectedModel === 'lightgbm' ? '/predict/lightgbm' 
                      : selectedModel === 'xgboost' ? '/predict/xgboost'
                      : '/predict/ensemble';
      
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const result = await response.json();
      
      if (selectedModel === 'ensemble') {
        setEnsemblePrediction({
          lightgbm_prediction: result.lightgbm_prediction,
          xgboost_prediction: result.xgboost_prediction,
          mean_prediction: result.mean_prediction,
        });
        setModel('Ensemble (LightGBM + XGBoost)');
      } else {
        setPrediction(result.predicted_median_house_value);
        setModel(selectedModel === 'lightgbm' ? 'LightGBM' : 'XGBoost');
      }
      
      toast.success(`Prediction complete using ${selectedModel === 'ensemble' ? 'Ensemble' : selectedModel}!`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to get prediction. Please check if the API is running.');
      console.error('Prediction error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div 
          className="absolute inset-0 z-0"
          style={{
            backgroundImage: `url(${californiaHero})`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            backgroundRepeat: 'no-repeat',
          }}
        >
          <div className="absolute inset-0 bg-gradient-to-b from-background/80 via-background/60 to-background"></div>
        </div>
        
        <div className="relative z-10 container mx-auto px-4 pt-16 pb-12 text-center">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent animate-fade-in">
            California Housing Price Predictor
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto animate-fade-in" style={{ animationDelay: '0.1s' }}>
            Enter the details of a house and get its predicted median value using advanced ML models.
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-12">
        <PredictionForm onPredict={handlePredict} isLoading={isLoading} />
        <PredictionResult 
          prediction={prediction} 
          ensemblePrediction={ensemblePrediction}
          model={model} 
        />
      </div>

      {/* Footer */}
      <footer className="mt-20 py-8 border-t border-border">
        <div className="container mx-auto px-4 text-center">
          <p className="text-sm text-muted-foreground">
            Developed with{" "}
            <span className="text-primary">❤</span>{" "}
            using advanced machine learning techniques
          </p>
          <p className="text-xs text-muted-foreground mt-2">
            © 2025 California Housing Price Predictor
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
