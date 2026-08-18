import { Link, useLocation } from "react-router-dom";

import type { ResultRouteState } from "../types/prediction";

export function ResultPage() {
  const location = useLocation();
  const state = location.state as ResultRouteState | null;

  if (!state?.prediction || !state.request) {
    return (
      <main className="centered-page">
        <div className="empty-card">
          <span className="empty-icon">₹</span>
          <p className="eyebrow">No saved estimate</p>
          <h1>Start with the property form.</h1>
          <p>Prediction details are kept only for the current navigation session.</p>
          <Link to="/" className="primary-button link-button">Create an estimate</Link>
        </div>
      </main>
    );
  }

  const { prediction, request } = state;
  return (
    <main className="result-page">
      <section className="result-shell">
        <div className="result-summary">
          <p className="eyebrow">Estimated property price</p>
          <h1>{prediction.formatted_price}</h1>
          <p className="exact-price">
            {new Intl.NumberFormat("en-IN", {
              style: "currency",
              currency: "INR",
              maximumFractionDigits: 0,
            }).format(prediction.predicted_price)}
          </p>
          <div className="confidence-note">
            <span aria-hidden="true">i</span>
            <p>{prediction.disclaimer}</p>
          </div>
          <Link to="/" className="primary-button link-button">Estimate another property</Link>
        </div>

        <div className="result-details">
          <p className="eyebrow">Estimate inputs</p>
          <h2>Property snapshot</h2>
          <dl>
            <Detail label="Location" value={request.location} />
            <Detail label="Carpet area" value={`${request.carpet_area_sqft.toLocaleString("en-IN")} sqft`} />
            <Detail label="Floor" value={`${request.floor_num} of ${request.total_floors}`} />
            <Detail label="Bathrooms" value={String(request.bathroom)} />
            <Detail label="Balconies" value={String(request.balcony)} />
            <Detail label="Parking" value={String(request.parking)} />
            <Detail label="Furnishing" value={request.furnishing} />
            <Detail label="Transaction" value={request.transaction} />
            <Detail label="Ownership" value={request.ownership} />
            <Detail label="Facing" value={request.facing} />
          </dl>
          <p className="model-version">Model version {prediction.model_version}</p>
        </div>
      </section>
    </main>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return <div><dt>{label}</dt><dd>{value}</dd></div>;
}
