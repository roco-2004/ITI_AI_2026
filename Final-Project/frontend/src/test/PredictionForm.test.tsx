import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { PredictionForm } from "../components/PredictionForm";
import type { PredictionResponse } from "../types/prediction";

const response: PredictionResponse = {
  predicted_price: 7_500_000,
  formatted_price: "₹75.00 Lakh",
  currency: "INR",
  model_version: "test",
  disclaimer: "Informational estimate only.",
};

function renderForm(onPredict = vi.fn().mockResolvedValue(response)) {
  const onSuccess = vi.fn();
  render(
    <PredictionForm
      locations={["mumbai", "Other"]}
      locationsLoading={false}
      locationsError={null}
      onRetryLocations={vi.fn()}
      onPredict={onPredict}
      onSuccess={onSuccess}
    />,
  );
  return { onPredict, onSuccess };
}

async function chooseLocation() {
  await userEvent.selectOptions(screen.getByLabelText("Location"), "mumbai");
}

describe("PredictionForm", () => {
  it("shows friendly validation for invalid area", async () => {
    renderForm();
    await chooseLocation();
    const area = screen.getByLabelText(/Carpet area/);
    await userEvent.clear(area);
    await userEvent.type(area, "0");
    await userEvent.click(screen.getByRole("button", { name: /Estimate property price/ }));
    expect(screen.getByText("Area must be between 100 and 20,000 sqft.")).toBeInTheDocument();
  });

  it("disables submit and shows a loading state", async () => {
    const pending = new Promise<PredictionResponse>(() => undefined);
    renderForm(vi.fn().mockReturnValue(pending));
    await chooseLocation();
    await userEvent.click(screen.getByRole("button", { name: /Estimate property price/ }));
    expect(screen.getByRole("button", { name: /Calculating estimate/ })).toBeDisabled();
  });

  it("submits numeric payload and reports success", async () => {
    const { onPredict, onSuccess } = renderForm();
    await chooseLocation();
    await userEvent.click(screen.getByRole("button", { name: /Estimate property price/ }));
    await waitFor(() => expect(onSuccess).toHaveBeenCalledOnce());
    expect(onPredict).toHaveBeenCalledWith(expect.objectContaining({ carpet_area_sqft: 1200 }));
  });

  it("shows API failure and a retry action", async () => {
    renderForm(vi.fn().mockRejectedValue(new Error("Prediction service unavailable")));
    await chooseLocation();
    await userEvent.click(screen.getByRole("button", { name: /Estimate property price/ }));
    expect(await screen.findByText("Prediction service unavailable")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry prediction" })).toBeInTheDocument();
  });
});
