import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { fetchLocations, predictHouse } from "../api/predictionClient";
import { PredictionForm } from "../components/PredictionForm";
import type { PredictionRequest, PredictionResponse } from "../types/prediction";

export function HomePage() {
  const navigate = useNavigate();
  const [locations, setLocations] = useState<string[]>([]);
  const [locationsLoading, setLocationsLoading] = useState(true);
  const [locationsError, setLocationsError] = useState<string | null>(null);

  const loadLocations = useCallback(async () => {
    setLocationsLoading(true);
    setLocationsError(null);
    try {
      const response = await fetchLocations();
      setLocations(response.locations);
    } catch {
      setLocationsError("Locations could not be loaded from the prediction service.");
    } finally {
      setLocationsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadLocations();
  }, [loadLocations]);

  const showResult = (prediction: PredictionResponse, request: PredictionRequest) => {
    navigate("/result", { state: { prediction, request } });
  };

  return (
    <main>
      <section className="hero">
        <div className="hero-content">
          <p className="eyebrow hero-eyebrow">Machine-learning estimate · India</p>
          <h1>Understand a home’s <em>estimated value</em> in seconds.</h1>
          <p className="hero-copy">
            Enter practical property details and receive an INR estimate from a reproducible model
            trained on Indian residential listings.
          </p>
          <div className="trust-row" aria-label="Model highlights">
            <div><strong>36,551</strong><span>clean listings</span></div>
            <div><strong>11</strong><span>property features</span></div>
            <div><strong>0.6045</strong><span>held-out R²</span></div>
          </div>
        </div>
        <div className="hero-visual" aria-hidden="true">
          <div className="sun" />
          <div className="building building-one"><i /><i /><i /><i /></div>
          <div className="building building-two"><i /><i /><i /><i /><i /><i /></div>
          <div className="building building-three"><i /><i /><i /></div>
          <div className="visual-card">
            <span>Indicative estimate</span>
            <strong>₹78.4 Lakh</strong>
            <small>Model-backed · INR</small>
          </div>
        </div>
      </section>

      <section className="estimator-section" id="estimate">
        <div className="section-intro">
          <p className="eyebrow">Price estimator</p>
          <h2>A clearer starting point for your property research.</h2>
          <p>
            The model combines area, location, floor details, amenities, and listing attributes.
            Results are transparent about uncertainty and intended for learning and comparison.
          </p>
          <ul className="feature-list">
            <li><span>01</span> Complete preprocessing pipeline</li>
            <li><span>02</span> Validation-selected boosting model</li>
            <li><span>03</span> Held-out test evaluation</li>
          </ul>
        </div>

        <PredictionForm
          locations={locations}
          locationsLoading={locationsLoading}
          locationsError={locationsError}
          onRetryLocations={() => void loadLocations()}
          onPredict={predictHouse}
          onSuccess={showResult}
        />
      </section>

      <section className="disclaimer-strip">
        <strong>Informational use only.</strong>
        <span>
          Estimates are not professional appraisals, transaction guarantees, or investment advice.
        </span>
      </section>
    </main>
  );
}
