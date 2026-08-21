import { Link } from "react-router-dom";

export function NotFoundPage() {
  return (
    <main className="centered-page">
      <section className="empty-card" aria-labelledby="not-found-title">
        <span className="error-code" aria-hidden="true">
          404
        </span>

        <p className="eyebrow">Page not found</p>

        <h1 id="not-found-title">
          This page is not available.
        </h1>

        <p>
          The address you entered does not match any page in the
          house price estimator.
        </p>

        <Link to="/" className="primary-button link-button">
          Return to estimator
        </Link>
      </section>
    </main>
  );
}
