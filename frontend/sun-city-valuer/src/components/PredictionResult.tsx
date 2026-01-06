import { Card, CardContent } from "@/components/ui/card";
import { TrendingUp } from "lucide-react";

interface EnsemblePrediction {
  lightgbm_prediction: number;
  xgboost_prediction: number;
  mean_prediction: number;
}

interface PredictionResultProps {
  prediction: number | null;
  ensemblePrediction: EnsemblePrediction | null;
  model: string | null;
}

export const PredictionResult = ({ prediction, ensemblePrediction, model }: PredictionResultProps) => {
  if (!prediction && !ensemblePrediction) return null;

  if (ensemblePrediction) {
    return (
      <div className="w-full max-w-4xl mx-auto mt-8 animate-fade-in space-y-4">
        <Card className="bg-gradient-to-br from-primary/10 to-secondary/10 border-primary/20 shadow-[var(--shadow-glow)]">
          <CardContent className="pt-6">
            <div className="flex items-center justify-center gap-4 mb-6">
              <div className="p-3 rounded-full bg-primary/20 animate-scale-in">
                <TrendingUp className="h-8 w-8 text-primary" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground font-medium">
                  Predicted using {model}
                </p>
                <p className="text-4xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                  ${ensemblePrediction.mean_prediction.toLocaleString()}
                </p>
                <p className="text-sm text-muted-foreground mt-1">
                  Mean Prediction
                </p>
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div className="p-4 rounded-lg bg-background/50 backdrop-blur-sm border border-primary/10">
                <p className="text-xs text-muted-foreground mb-1">LightGBM Model</p>
                <p className="text-2xl font-bold text-primary">
                  ${ensemblePrediction.lightgbm_prediction.toLocaleString()}
                </p>
              </div>
              <div className="p-4 rounded-lg bg-background/50 backdrop-blur-sm border border-secondary/10">
                <p className="text-xs text-muted-foreground mb-1">XGBoost Model</p>
                <p className="text-2xl font-bold text-secondary">
                  ${ensemblePrediction.xgboost_prediction.toLocaleString()}
                </p>
              </div>
            </div>

            <div className="p-4 rounded-lg bg-background/50 backdrop-blur-sm">
              <p className="text-sm text-muted-foreground text-center">
                Ensemble prediction combines both LightGBM and XGBoost models for improved accuracy.
                Results may vary based on market conditions and specific property features.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="w-full max-w-4xl mx-auto mt-8 animate-fade-in">
      <Card className="bg-gradient-to-br from-primary/10 to-secondary/10 border-primary/20 shadow-[var(--shadow-glow)]">
        <CardContent className="pt-6">
          <div className="flex items-center justify-center gap-4 mb-4">
            <div className="p-3 rounded-full bg-primary/20 animate-scale-in">
              <TrendingUp className="h-8 w-8 text-primary" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground font-medium">
                Predicted using {model}
              </p>
              <p className="text-4xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                ${prediction.toLocaleString()}
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                Median House Value
              </p>
            </div>
          </div>
          <div className="mt-4 p-4 rounded-lg bg-background/50 backdrop-blur-sm">
            <p className="text-sm text-muted-foreground text-center">
              This prediction is based on advanced machine learning models trained on California housing data.
              Results may vary based on market conditions and specific property features.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
