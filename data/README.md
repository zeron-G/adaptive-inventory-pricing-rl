# Data Notes

This repository does not store raw Kaggle data. Download datasets locally as needed.

## Primary Dataset

M5 Forecasting - Accuracy:

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Expected key files:

- `sales_train_validation.csv` or `sales_train_evaluation.csv`
- `sell_prices.csv`
- `calendar.csv`

Suggested local layout after download:

```text
data/
  raw/
    m5/
      sales_train_validation.csv
      sell_prices.csv
      calendar.csv
  processed/
```

## Kaggle CLI Example

If Kaggle credentials are configured:

```powershell
kaggle competitions download -c m5-forecasting-accuracy -p data/raw/m5
Expand-Archive -Path data/raw/m5/m5-forecasting-accuracy.zip -DestinationPath data/raw/m5
```

## Prototype Datasets

Store Item Demand Forecasting Dataset:

https://www.kaggle.com/datasets/dhrubangtalukdar/store-item-demand-forecasting-dataset

Retail Sales Promotions and Demand Forecasting:

https://www.kaggle.com/datasets/jayjoshi37/retail-sales-promotions-and-demand-forecasting

These smaller datasets can be used to test the simulator structure before processing M5.
