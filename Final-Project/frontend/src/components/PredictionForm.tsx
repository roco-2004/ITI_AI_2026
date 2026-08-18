import { type FormEvent, useMemo, useState } from "react";

import type {
  Facing,
  Furnishing,
  Ownership,
  PredictionRequest,
  PredictionResponse,
  Transaction,
} from "../types/prediction";

interface PredictionFormProps {
  locations: string[];
  locationsLoading: boolean;
  locationsError: string | null;
  onRetryLocations: () => void;
  onPredict: (request: PredictionRequest) => Promise<PredictionResponse>;
  onSuccess: (prediction: PredictionResponse, request: PredictionRequest) => void;
}

type FormValues = Record<keyof PredictionRequest, string>;
type FormErrors = Partial<Record<keyof PredictionRequest, string>>;

const initialValues: FormValues = {
  location: "",
  carpet_area_sqft: "1200",
  floor_num: "3",
  total_floors: "10",
  bathroom: "2",
  balcony: "1",
  parking: "1",
  furnishing: "Semi-Furnished",
  transaction: "Resale",
  ownership: "Freehold",
  facing: "East",
};

const numericFields = [
  "carpet_area_sqft",
  "floor_num",
  "total_floors",
  "bathroom",
  "balcony",
  "parking",
] as const;

function validate(values: FormValues): FormErrors {
  const errors: FormErrors = {};
  if (!values.location) errors.location = "Choose a location.";

  for (const field of numericFields) {
    if (values[field].trim() === "" || !Number.isFinite(Number(values[field]))) {
      errors[field] = "Enter a valid number.";
    }
  }

  const area = Number(values.carpet_area_sqft);
  if (Number.isFinite(area) && (area < 100 || area > 20_000)) {
    errors.carpet_area_sqft = "Area must be between 100 and 20,000 sqft.";
  }
  const floor = Number(values.floor_num);
  const totalFloors = Number(values.total_floors);
  if (floor < -1 || floor > 100) errors.floor_num = "Floor must be between -1 and 100.";
  if (totalFloors < 1 || totalFloors > 100) {
    errors.total_floors = "Total floors must be between 1 and 100.";
  }
  if (floor > totalFloors) errors.floor_num = "Floor cannot exceed total floors.";

  const bounded: Array<[keyof FormValues, number, number]> = [
    ["bathroom", 1, 11],
    ["balcony", 0, 11],
    ["parking", 0, 10],
  ];
  for (const [field, minimum, maximum] of bounded) {
    const value = Number(values[field]);
    if (Number.isFinite(value) && (!Number.isInteger(value) || value < minimum || value > maximum)) {
      errors[field] = `Use a whole number from ${minimum} to ${maximum}.`;
    }
  }
  return errors;
}

export function PredictionForm({
  locations,
  locationsLoading,
  locationsError,
  onRetryLocations,
  onPredict,
  onSuccess,
}: PredictionFormProps) {
  const [values, setValues] = useState<FormValues>(initialValues);
  const [errors, setErrors] = useState<FormErrors>({});
  const [apiError, setApiError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const sortedLocations = useMemo(
    () => [...locations].sort((a, b) => a.localeCompare(b)),
    [locations],
  );

  const update = (field: keyof FormValues, value: string) => {
    setValues((current) => ({ ...current, [field]: value }));
    setErrors((current) => ({ ...current, [field]: undefined }));
  };

  const submit = async (event?: FormEvent) => {
    event?.preventDefault();
    const nextErrors = validate(values);
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length > 0) return;

    const payload: PredictionRequest = {
      location: values.location,
      carpet_area_sqft: Number(values.carpet_area_sqft),
      floor_num: Number(values.floor_num),
      total_floors: Number(values.total_floors),
      bathroom: Number(values.bathroom),
      balcony: Number(values.balcony),
      parking: Number(values.parking),
      furnishing: values.furnishing as Furnishing,
      transaction: values.transaction as Transaction,
      ownership: values.ownership as Ownership,
      facing: values.facing as Facing,
    };

    setSubmitting(true);
    setApiError(null);
    try {
      const prediction = await onPredict(payload);
      onSuccess(prediction, payload);
    } catch (error) {
      setApiError(error instanceof Error ? error.message : "Prediction failed. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form className="prediction-form" onSubmit={submit} noValidate>
      <div className="form-heading">
        <div>
          <p className="eyebrow">Property details</p>
          <h2>Describe the home</h2>
        </div>
        <span className="step-badge">All fields required</span>
      </div>

      {locationsError && (
        <div className="alert alert-error" role="alert">
          <span>{locationsError}</span>
          <button type="button" className="text-button" onClick={onRetryLocations}>
            Retry locations
          </button>
        </div>
      )}

      <div className="form-grid">
        <label className="field field-wide">
          <span>Location</span>
          <select
            value={values.location}
            onChange={(event) => update("location", event.target.value)}
            disabled={locationsLoading || locations.length === 0}
            aria-invalid={Boolean(errors.location)}
          >
            <option value="">{locationsLoading ? "Loading locations…" : "Select a location"}</option>
            {sortedLocations.map((location) => (
              <option key={location} value={location}>
                {location === "Other" ? "Other / unlisted" : location}
              </option>
            ))}
          </select>
          {errors.location && <small className="field-error">{errors.location}</small>}
        </label>

        <NumberField
          label="Carpet area"
          suffix="sqft"
          value={values.carpet_area_sqft}
          error={errors.carpet_area_sqft}
          onChange={(value) => update("carpet_area_sqft", value)}
          min="100"
          max="20000"
        />
        <NumberField
          label="Current floor"
          value={values.floor_num}
          error={errors.floor_num}
          onChange={(value) => update("floor_num", value)}
          min="-1"
          max="100"
        />
        <NumberField
          label="Total floors"
          value={values.total_floors}
          error={errors.total_floors}
          onChange={(value) => update("total_floors", value)}
          min="1"
          max="100"
        />
        <NumberField
          label="Bathrooms"
          value={values.bathroom}
          error={errors.bathroom}
          onChange={(value) => update("bathroom", value)}
          min="1"
          max="11"
        />
        <NumberField
          label="Balconies"
          value={values.balcony}
          error={errors.balcony}
          onChange={(value) => update("balcony", value)}
          min="0"
          max="11"
        />
        <NumberField
          label="Parking spaces"
          value={values.parking}
          error={errors.parking}
          onChange={(value) => update("parking", value)}
          min="0"
          max="10"
        />

        <SelectField
          label="Furnishing"
          value={values.furnishing}
          options={["Furnished", "Semi-Furnished", "Unfurnished"]}
          onChange={(value) => update("furnishing", value)}
        />
        <SelectField
          label="Transaction"
          value={values.transaction}
          options={["Resale", "New Property", "Other"]}
          onChange={(value) => update("transaction", value)}
        />
        <SelectField
          label="Ownership"
          value={values.ownership}
          options={["Freehold", "Leasehold", "Co-operative Society", "Power Of Attorney"]}
          onChange={(value) => update("ownership", value)}
        />
        <SelectField
          label="Facing"
          value={values.facing}
          options={[
            "East",
            "North",
            "North - East",
            "North - West",
            "South",
            "South - East",
            "South -West",
            "West",
          ]}
          onChange={(value) => update("facing", value)}
        />
      </div>

      {apiError && (
        <div className="alert alert-error" role="alert">
          <span>{apiError}</span>
          <button type="button" className="text-button" onClick={() => void submit()}>
            Retry prediction
          </button>
        </div>
      )}

      <button
        type="submit"
        className="primary-button"
        disabled={submitting || locationsLoading || locations.length === 0}
      >
        {submitting ? <span className="spinner" aria-hidden="true" /> : <span>₹</span>}
        {submitting ? "Calculating estimate…" : "Estimate property price"}
      </button>
      <p className="form-note">Your inputs are sent only to the configured prediction API.</p>
    </form>
  );
}

interface NumberFieldProps {
  label: string;
  value: string;
  error?: string;
  onChange: (value: string) => void;
  min: string;
  max: string;
  suffix?: string;
}

function NumberField({ label, value, error, onChange, min, max, suffix }: NumberFieldProps) {
  const id = label.toLowerCase().replaceAll(" ", "-");
  return (
    <label className="field" htmlFor={id}>
      <span>{label}</span>
      <div className="input-with-suffix">
        <input
          id={id}
          type="number"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          min={min}
          max={max}
          step="1"
          aria-invalid={Boolean(error)}
        />
        {suffix && <span>{suffix}</span>}
      </div>
      {error && <small className="field-error">{error}</small>}
    </label>
  );
}

interface SelectFieldProps {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}

function SelectField({ label, value, options, onChange }: SelectFieldProps) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}
