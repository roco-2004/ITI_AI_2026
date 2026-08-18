export type Furnishing = "Furnished" | "Semi-Furnished" | "Unfurnished";
export type Transaction = "New Property" | "Other" | "Resale";
export type Ownership =
  | "Co-operative Society"
  | "Freehold"
  | "Leasehold"
  | "Power Of Attorney";
export type Facing =
  | "East"
  | "North"
  | "North - East"
  | "North - West"
  | "South"
  | "South - East"
  | "South -West"
  | "West";

export interface PredictionRequest {
  location: string;
  carpet_area_sqft: number;
  floor_num: number;
  total_floors: number;
  bathroom: number;
  balcony: number;
  parking: number;
  furnishing: Furnishing;
  transaction: Transaction;
  ownership: Ownership;
  facing: Facing;
}

export interface PredictionResponse {
  predicted_price: number;
  formatted_price: string;
  currency: "INR";
  model_version: string;
  disclaimer: string;
}

export interface LocationsResponse {
  locations: string[];
  other_label: string;
}

export interface ResultRouteState {
  prediction: PredictionResponse;
  request: PredictionRequest;
}
