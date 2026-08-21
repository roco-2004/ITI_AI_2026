import { Link, useLocation } from "react-router-dom";

import type { ResultRouteState } from "../types/prediction";

export function ResultPage() {
  const { state } = useLocation();
  const result = state as ResultRouteState | null;

  if (!result?.prediction || !result.request) {
    return <MissingResult />;
  }

  const { prediction, request } = result;

  const formattedExactPrice = new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(prediction.predicted_price);

  return (
    <main className="result-page">
      <section className="result-shell">
        <article className="result-summary">
          <p className="eyebrow">Estimated property price</p>

          <h1>{prediction.formatted_price}</h1>

          <p className="exact-price">{formattedExactPrice}</p>

          <div className="confidence-note">
            <span aria-hidden="true">i</span>
            <p>{prediction.disclaimer}</p>
          </div>

          <Link to="/" className="primary-button link-button">
            Estimate another property
          </Link>
        </article>

        <article className="result-details">
          <p className="eyebrow">Estimate inputs</p>
          <h2>Property snapshot</h2>

          <dl>
            <Detail
              label="Location"
              value={request.location}
            />
            <Detail
              label="Carpet area"
              value={`${request.carpet_area_sqft.toLocaleString("en-IN")} sqft`}
            />
            <Detail
              label="Floor"
              value={`${request.floor_num} of ${request.total_floors}`}
            />
            <Detail
              label="Bathrooms"
              value={String(request.bathroom)}
            />
            <Detail
              label="Balconies"
              value={String(request.balcony)}
            />
            <Detail
              label="Parking"
              value={String(request.parking)}
            />
            <Detail
              label="Furnishing"
              value={request.furnishing}
            />
            <Detail
              label="Transaction"
              value={request.transaction}
            />
            <Detail
              label="Ownership"
              value={request.ownership}
            />
            <Detail
              label="Facing"
              value={request.facing}
            />
          </dl>

          <p className="model-version">
            Model version {prediction.model_version}
          </p>
        </article>
      </section>
    </main>
  );
}

function MissingResult() {
  return (
    <main className="centered-page">
      <section className="empty-card">
        <span className="empty-icon" aria-hidden="true">
          ₹
        </span>

        <p className="eyebrow">No saved estimate</p>

        <h1>Start with the property form.</h1>

        <p>
          Prediction details are available only during the current
          navigation session.
        </p>

        <Link to="/" className="primary-button link-button">
          Create an estimate
        </Link>
      </section>
    </main>
  );
}

interface DetailProps {
  label: string;
  value: string;
}

function Detail({ label, value }: DetailProps) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}
