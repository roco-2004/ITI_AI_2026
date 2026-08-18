import { Link } from "react-router-dom";

export function NotFoundPage() {
  return (
    <main className="centered-page">
      <div className="empty-card">
        <span className="error-code">404</span>
        <p className="eyebrow">Page not found</p>
        <h1>This address has no listing.</h1>
        <p>Return to the estimator and start a new property estimate.</p>
        <Link to="/" className="primary-button link-button">Back to estimator</Link>
      </div>
    </main>
  );
}
