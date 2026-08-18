# Dataset download

The raw dataset is intentionally excluded from Git because it is approximately 101.23 MiB and contains full property-listing text. Download it from the Kaggle dataset [House Price by Juhi Bhojani](https://www.kaggle.com/datasets/juhibhojani/house-price).

From the `Final-Project` directory, either download and extract the archive manually or use the Kaggle CLI:

```powershell
python -m pip install kaggle
kaggle datasets download -d juhibhojani/house-price -p notebooks/data --unzip
```

The notebook expects this exact relative path:

```text
notebooks/data/house_prices.csv
```

The locally audited file had:

- Size: `106,149,815` bytes
- SHA-256: `ED1E6A1A2D1158F458CEF164F7E977F2C32A1EE392AD218037C11851814EF1E3`
- Shape: `187,531 rows × 21 columns`

Kaggle may publish a revised file in the future. A checksum difference should trigger a fresh audit; it is not automatically evidence of corruption. Never commit the CSV, Kaggle credentials, or downloaded archives.
