# House Price Dataset Audit

## Provenance and integrity

- Source: [Kaggle — House Price by Juhi Bhojani](https://www.kaggle.com/datasets/juhibhojani/house-price)
- Audited filename: `house_prices.csv`
- File size: 106,149,815 bytes (101.23 MiB)
- SHA-256: `ED1E6A1A2D1158F458CEF164F7E977F2C32A1EE392AD218037C11851814EF1E3`
- Shape: 187,531 rows × 21 columns
- In-memory pandas footprint: 270,979,135 bytes (258.43 MiB)
- Audit environment: Python 3.11.9 and pandas 2.2.3

The audit used aggregate statistics and format frequencies. It did not print titles, descriptions, or full listing records.

## Column-level audit

| Column | dtype | Missing | Missing % | Unique non-null | Assessment |
| --- | --- | ---: | ---: | ---: | --- |
| Index | int64 | 0 | 0.0000 | 187,531 | Row identifier; excluded |
| Title | object | 0 | 0.0000 | 32,446 | High-cardinality listing text; privacy/leakage risk; excluded |
| Description | object | 3,023 | 1.6120 | 65,634 | Free text may contain contact or price details; excluded |
| Amount(in rupees) | object | 0 | 0.0000 | 1,561 | Target source after parsing |
| Price (in rupees) | float64 | 17,665 | 9.4198 | 10,958 | Price-per-area derivative; target leakage; excluded |
| location | object | 0 | 0.0000 | 81 | Categorical feature with train-only rare grouping |
| Carpet Area | object | 80,673 | 43.0185 | 2,758 | Primary area source after unit parsing |
| Status | object | 615 | 0.3279 | 1 | Constant (`Ready to Move`); excluded |
| Floor | object | 7,077 | 3.7738 | 947 | Parsed into current and total floors |
| Transaction | object | 83 | 0.0443 | 4 | Categorical; rental rows excluded from sale-price modeling |
| Furnishing | object | 2,897 | 1.5448 | 3 | Categorical feature |
| facing | object | 70,233 | 37.4514 | 8 | Categorical feature with pipeline imputation |
| overlooking | object | 81,436 | 43.4254 | 19 | Inconsistent multi-value strings; excluded |
| Society | object | 109,678 | 58.4853 | 10,376 | High-cardinality property identifier; excluded |
| Bathroom | object | 828 | 0.4415 | 11 | Parsed numeric feature |
| Balcony | object | 48,935 | 26.0944 | 11 | Parsed numeric feature |
| Car Parking | object | 103,357 | 55.1146 | 229 | Parsed numeric feature; communal counts above 10 treated missing |
| Ownership | object | 65,517 | 34.9366 | 4 | Categorical feature |
| Super Area | object | 107,685 | 57.4225 | 2,976 | Audited but excluded to keep input semantics consistently carpet area |
| Dimensions | float64 | 187,531 | 100.0000 | 0 | Empty constant column; excluded |
| Plot Area | float64 | 187,531 | 100.0000 | 0 | Empty constant column; excluded |

## Duplicate analysis

There are zero exact full-row duplicates because `Index` is unique. Excluding only `Index`, 119,339 rows duplicate every other source field. These are treated as duplicate source listings and removed before splitting. Deduplication retains `Title`, `Description`, and `Society` in the comparison key, so separate listings are not collapsed merely because their model features and prices happen to match.

## Target formats and validity

`Amount(in rupees)` contains values such as `85 Lac`, `1.75 Cr`, and `Call for Price`. The reusable parser handles rupee symbols, commas, `Lac`/`Lakh`, `Cr`/`Crore`, and plain numeric values.

- Parsed positive targets: 177,847
- Unavailable or invalid targets: 9,684 (all dominated by `Call for Price`)
- Parsed minimum: ₹100,000
- Parsed median: ₹7,800,000
- Parsed 99th percentile: ₹70,000,000
- Parsed maximum: ₹14,003,000,000

The extreme right tail is not silently removed from the full dataset. Price-per-square-foot outlier limits are learned from the training split only and then used to filter training rows, preserving an untouched, representative test set.

## Area formats and units

`Carpet Area` contains 100,428 `sqft`, 5,526 `sqyrd`, 894 `sqm`, 3 `marla`, 2 `acre`, 2 `kanal`, and one each of `ground`, `bigha`, and `cent`. `Super Area` additionally contains rare `hectare`, `biswa`, and `aankadam` values.

Supported deterministic conversions include:

- 1 square metre = 10.7639 square feet
- 1 square yard = 9 square feet
- 1 acre = 43,560 square feet
- 1 hectare = 107,639.104 square feet
- 1 cent = 435.6 square feet
- 1 marla = 272.25 square feet
- 1 kanal = 5,445 square feet
- 1 ground = 2,400 square feet

`bigha`, `biswa`, and `aankadam` are explicitly recognized but returned as missing because their size is regional or insufficiently specified. This avoids fabricating precise conversions. The model uses only parsed `Carpet Area`, rather than mixing carpet and super-area semantics.

- Valid parsed carpet areas: 106,857
- Missing/unsupported carpet areas: 80,674
- Parsed values outside the documented 100–20,000 sqft residential plausibility range: 1,755
- Parsed median: 1,064 sqft
- Parsed maximum before plausibility filtering: 65,340,000 sqft

## Floor and count formats

Floor values include `2 out of 4`, `Ground out of 4`, `Lower Basement`, and `Upper Basement`. Ground is encoded as 0 and basement as -1; both current and total floors are retained. One current-floor and two total-floor values exceed the 100-floor plausibility ceiling and are treated as missing.

Bathrooms and balconies contain numeric strings plus `> 10`, which is represented as 11. Parking includes values such as `1 Covered` and `2 Open`, plus implausible communal counts such as `402 Covered`; counts above 10 are treated as missing.

## Modeling population

Deterministic cleaning produces:

| Decision | Rows affected |
| --- | ---: |
| Raw rows | 187,531 |
| Duplicate source listings removed | 119,339 |
| Unusable target rows removed after deduplication | 2,937 |
| Missing/unsupported/implausible carpet-area rows removed | 28,698 |
| Rental transaction rows removed | 1 |
| Final cleaned modeling rows | 36,556 |

The final cleaned table has 11 input features plus the target. Missing values remain in floor, bathroom, balcony, parking, furnishing, transaction, ownership, and facing so they can be imputed inside the fitted sklearn pipeline.

## Leakage, privacy, and dropped columns

- `Amount(in rupees)` defines the target and is not a feature.
- `Price (in rupees)` behaves as a price-per-area derivative and would leak the target.
- No price-per-square-foot feature is used.
- `Index` is an identifier.
- `Title` and `Description` are excluded because they may reveal price, identity, contact details, or listing-specific text.
- `Society` is a high-cardinality property identifier with 58.49% missingness.
- `Status`, `Dimensions`, and `Plot Area` are constant or empty.
- `overlooking` contains inconsistent multi-value ordering and is not needed for the compact application schema.
- `Super Area` is excluded so the UI's carpet-area input matches the trained feature definition.

No raw dataset content is committed. The ignored local CSV is required only for notebook retraining; the application uses the committed model artifact.
