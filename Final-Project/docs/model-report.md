# Model Report

## Objective

Estimate Indian residential listing sale prices in rupees from a compact set of property attributes. The result is an educational informational estimate, not a professional appraisal or investment recommendation.

## Data and cleaning decisions

The audited Kaggle file contains 187,531 rows and 21 columns. Exact source duplicates are identified across all columns except the unique `Index`, removing 119,339 duplicated listing records. Rows without a usable sale-price target or a supported 100–20,000 sqft carpet area are excluded. One rental row is excluded because rent and sale prices are different targets.

Five records outside the fixed ₹100–₹500,000 per-square-foot integrity range are removed from the full population as obvious unit or zero corruption. This rule is intentionally broad and predetermined. A tighter 1st/99th-percentile range of ₹2,234.21–₹35,495.52 per square foot is learned from the training split only and applied only to model-fitting rows.

The final modeling population has 36,551 unique cleaned listings. The split is deterministic with seed 42:

- Initial training: 21,930 rows
- Validation: 7,310 rows
- Untouched test: 7,311 rows
- Final train-plus-validation rows after applying the fixed training-derived outlier limits: 28,649

## Features

Numeric features are carpet area, current floor, total floors, bathrooms, balconies, and parking. Categorical features are location, furnishing, transaction, ownership, and facing.

The following columns are deliberately excluded:

- `Amount(in rupees)`: target source
- `Price (in rupees)`: target-derived price-per-area leakage
- `Index`: identifier
- `Title` and `Description`: listing-specific text with price, identity, and privacy risk
- `Society`: high-cardinality property identifier with substantial missingness
- `Status`, `Dimensions`, and `Plot Area`: constant or empty
- `overlooking`: inconsistent multi-value strings
- `Super Area`: excluded so the model and user interface consistently use carpet area

Missing numeric and categorical values are imputed inside the pipeline. Location frequency is learned during pipeline fitting; rare and unseen locations map to `Other`. No reusable category rule is learned from validation or test data.

## Experiment comparison

### Validation results

| Model | MAE (INR) | MAE (lakh) | RMSE (INR) | RMSE (lakh) | R² | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Histogram gradient boosting, log target | 3,061,149 | 30.6115 | 9,989,151 | 99.8915 | 0.6073 | 3.1074 |
| Ridge regression | 4,270,159 | 42.7016 | 11,785,090 | 117.8509 | 0.4534 | 0.1150 |
| Dummy median | 6,647,547 | 66.4755 | 16,483,930 | 164.8393 | -0.0693 | 0.1141 |

### Five-fold cross-validation

Five shuffled folds use a fixed 20,000-row training-only sample to limit CPU and memory cost.

| Model | Mean MAE (INR) | MAE std. (INR) | Mean RMSE (INR) | Mean R² |
| --- | ---: | ---: | ---: | ---: |
| Histogram gradient boosting, log target | 2,520,994 | 64,412 | 5,347,087 | 0.7782 |
| Ridge regression | 3,546,915 | 51,146 | 6,027,687 | 0.7163 |
| Dummy median | 5,927,124 | 154,724 | 11,855,390 | -0.0966 |

## Selected model

Histogram gradient boosting with a log-transformed target is selected because it has the lowest predeclared validation MAE, the strongest validation R², and the best five-fold training-only CV summary. It captures non-linear area, location, and property interactions while training quickly and producing a compact artifact.

The exported object is a full scikit-learn pipeline containing numeric imputation and scaling, fitted-only rare-location grouping, categorical imputation and one-hot encoding, target transformation, and the regressor.

## One-time untouched test result

| Metric | Result |
| --- | ---: |
| MAE | ₹3,102,652.55 (31.0265 lakh) |
| RMSE | ₹10,924,915.85 (109.2492 lakh) |
| R² | 0.6045 |
| Total prediction time for 7,311 rows | 0.1729 seconds |

These are original-scale held-out test metrics, not training metrics. The validation-to-test change is reported without retuning.

## Error and interpretation analysis

The median absolute test error is ₹1,220,786. The 90th percentile is ₹6,319,956, the 95th percentile is ₹10,365,690, and the 99th percentile is ₹30,067,180. The maximum error is ₹411,713,300, demonstrating that luxury and unusual listings remain difficult.

Permutation importance on a fixed 2,000-row held-out sample ranks carpet area and location as the dominant inputs, followed by bathrooms and total floors. This analysis is performed after the one-time evaluation and is not used for model selection or tuning. Importance is predictive sensitivity, not causation.

## Serialization and reproducibility

- Artifact: `models/house_price.pkl`
- Artifact size: 445,708 bytes (0.425 MiB)
- Python: 3.11.9
- pandas: 2.2.3
- NumPy: 2.1.3
- scikit-learn: 1.6.1
- Dataset SHA-256: `ED1E6A1A2D1158F458CEF164F7E977F2C32A1EE392AD218037C11851814EF1E3`

The executed notebook reloads the artifact and obtains an identical prediction for a real held-out sample.

## Limitations

- The dataset contains advertised prices rather than confirmed transaction prices.
- Historical listings may not reflect current market conditions.
- Optional features have substantial missingness, especially parking and facing.
- Rare locations and luxury properties have less support and higher uncertainty.
- Deduplication relies on exact source-field equality and cannot identify every near-duplicate.
- Regional area units without a reliable conversion are rejected rather than guessed.
- No geospatial coordinates, property age, amenities, or market-time features are available.
- The application is an educational demonstration and is not production-grade.
