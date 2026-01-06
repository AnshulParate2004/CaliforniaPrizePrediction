import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Loader2 } from "lucide-react";

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

interface PredictionFormProps {
  onPredict: (data: FormData, model: 'lightgbm' | 'xgboost' | 'ensemble') => void;
  isLoading: boolean;
}

export const PredictionForm = ({ onPredict, isLoading }: PredictionFormProps) => {
  const [formData, setFormData] = useState<FormData>({
    longitude: "-122.23",
    latitude: "37.88",
    housingMedianAge: "41",
    totalRooms: "880",
    totalBedrooms: "129",
    population: "322",
    households: "126",
    medianIncome: "8.3252",
    oceanProximity: "NEAR BAY",
    bedroomRatio: "0.146",
    householdRooms: "6.984",
  });

  const handleInputChange = (field: keyof FormData, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handlePredict = (model: 'lightgbm' | 'xgboost' | 'ensemble') => {
    onPredict(formData, model);
  };

  return (
    <Card className="w-full max-w-4xl mx-auto backdrop-blur-sm bg-card/95 shadow-[var(--shadow-soft)]">
      <CardHeader>
        <CardTitle className="text-2xl font-bold">House Details</CardTitle>
        <CardDescription>Enter the property information for price prediction</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <Label htmlFor="longitude">Longitude</Label>
            <Input
              id="longitude"
              type="number"
              step="0.01"
              value={formData.longitude}
              onChange={(e) => handleInputChange("longitude", e.target.value)}
              placeholder="-122.23"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="latitude">Latitude</Label>
            <Input
              id="latitude"
              type="number"
              step="0.01"
              value={formData.latitude}
              onChange={(e) => handleInputChange("latitude", e.target.value)}
              placeholder="37.88"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="housingMedianAge">Housing Median Age</Label>
            <Input
              id="housingMedianAge"
              type="number"
              value={formData.housingMedianAge}
              onChange={(e) => handleInputChange("housingMedianAge", e.target.value)}
              placeholder="41"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="totalRooms">Total Rooms</Label>
            <Input
              id="totalRooms"
              type="number"
              value={formData.totalRooms}
              onChange={(e) => handleInputChange("totalRooms", e.target.value)}
              placeholder="880"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="totalBedrooms">Total Bedrooms</Label>
            <Input
              id="totalBedrooms"
              type="number"
              value={formData.totalBedrooms}
              onChange={(e) => handleInputChange("totalBedrooms", e.target.value)}
              placeholder="129"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="population">Population</Label>
            <Input
              id="population"
              type="number"
              value={formData.population}
              onChange={(e) => handleInputChange("population", e.target.value)}
              placeholder="322"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="households">Households</Label>
            <Input
              id="households"
              type="number"
              value={formData.households}
              onChange={(e) => handleInputChange("households", e.target.value)}
              placeholder="126"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="medianIncome">Median Income (in $10k)</Label>
            <Input
              id="medianIncome"
              type="number"
              step="0.0001"
              value={formData.medianIncome}
              onChange={(e) => handleInputChange("medianIncome", e.target.value)}
              placeholder="8.3252"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="oceanProximity">Ocean Proximity</Label>
            <Select value={formData.oceanProximity} onValueChange={(value) => handleInputChange("oceanProximity", value)}>
              <SelectTrigger id="oceanProximity">
                <SelectValue placeholder="Select proximity" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="NEAR BAY">Near Bay</SelectItem>
                <SelectItem value="<1H OCEAN">Less than 1 Hour to Ocean</SelectItem>
                <SelectItem value="INLAND">Inland</SelectItem>
                <SelectItem value="NEAR OCEAN">Near Ocean</SelectItem>
                <SelectItem value="ISLAND">Island</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="bedroomRatio">Bedroom Ratio</Label>
            <Input
              id="bedroomRatio"
              type="number"
              step="0.001"
              value={formData.bedroomRatio}
              onChange={(e) => handleInputChange("bedroomRatio", e.target.value)}
              placeholder="0.146"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="householdRooms">Household Rooms</Label>
            <Input
              id="householdRooms"
              type="number"
              step="0.001"
              value={formData.householdRooms}
              onChange={(e) => handleInputChange("householdRooms", e.target.value)}
              placeholder="6.984"
            />
          </div>
        </div>

        <div className="flex flex-col gap-4 mt-8">
          <div className="flex flex-col sm:flex-row gap-4">
            <Button
              onClick={() => handlePredict('lightgbm')}
              disabled={isLoading}
              className="flex-1 bg-gradient-to-r from-primary to-primary-glow hover:shadow-[var(--shadow-glow)] transition-all duration-300"
              size="lg"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Predicting...
                </>
              ) : (
                "Predict with LightGBM"
              )}
            </Button>

            <Button
              onClick={() => handlePredict('xgboost')}
              disabled={isLoading}
              variant="secondary"
              className="flex-1 bg-gradient-to-r from-secondary to-accent hover:shadow-[var(--shadow-soft)] transition-all duration-300"
              size="lg"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Predicting...
                </>
              ) : (
                "Predict with XGBoost"
              )}
            </Button>
          </div>
          
          <Button
            onClick={() => handlePredict('ensemble')}
            disabled={isLoading}
            variant="outline"
            className="w-full hover:bg-gradient-to-r hover:from-primary/10 hover:to-secondary/10 transition-all duration-300"
            size="lg"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                Predicting...
              </>
            ) : (
              "🔄 Predict with Ensemble (Both Models)"
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};
