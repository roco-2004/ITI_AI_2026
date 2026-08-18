import { BrowserRouter, Link, Route, Routes } from "react-router-dom";

import { HomePage } from "./pages/HomePage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { ResultPage } from "./pages/ResultPage";

export default function App() {
  return (
    <BrowserRouter>
      <header className="site-header">
        <Link to="/" className="brand" aria-label="India House Price Predictor home">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span><strong>India House</strong><small>Price Predictor</small></span>
        </Link>
        <a href="/#estimate" className="header-link">Create estimate <span>↗</span></a>
      </header>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/result" element={<ResultPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
      <footer>
        <span>India House Price Predictor</span>
        <span>ITI Artificial Intelligence Portfolio · Educational use</span>
      </footer>
    </BrowserRouter>
  );
}
