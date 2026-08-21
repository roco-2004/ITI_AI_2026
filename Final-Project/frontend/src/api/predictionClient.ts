import type {
  LocationsResponse,
  PredictionRequest,
  PredictionResponse,
} from "../types/prediction";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");

async function requestJson<T>(
  endpoint: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
  });

  if (!response.ok) {
    let errorMessage = `Request failed with status ${response.status}`;

    try {
      const data = (await response.json()) as {
        detail?: string;
      };

      if (data.detail) {
        errorMessage = data.detail;
      }
    } catch {
      // Use the default HTTP status message when the response is not JSON.
    }

    throw new Error(errorMessage);
  }

  return (await response.json()) as T;
}

export const fetchLocations = async (): Promise<LocationsResponse> => {
  return requestJson<LocationsResponse>("/api/locations");
};

export const predictHouse = async (
  payload: PredictionRequest,
): Promise<PredictionResponse> => {
  return requestJson<PredictionResponse>("/api/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
};
