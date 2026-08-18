import type {
  LocationsResponse,
  PredictionRequest,
  PredictionResponse,
} from "../types/prediction";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // Keep the status-based message when a response has no JSON body.
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export function fetchLocations(): Promise<LocationsResponse> {
  return requestJson<LocationsResponse>("/api/locations");
}

export function predictHouse(payload: PredictionRequest): Promise<PredictionResponse> {
  return requestJson<PredictionResponse>("/api/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
