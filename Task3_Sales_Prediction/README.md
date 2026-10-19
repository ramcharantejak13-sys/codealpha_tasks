# Task 4: Sales Prediction Using Python

## Objective
Predict product sales from advertising expenditure on TV, radio, and newspaper channels using a regression model.

## Dataset
Download the Advertising dataset from:
https://www.kaggle.com/datasets/ashydv/advertising-dataset

Save the CSV as `data/advertising.csv`. It should contain columns similar to `TV`, `Radio`, `Newspaper`, and `Sales`.

## Run
```bash
pip install -r requirements.txt
python Task4_Sales_Prediction/sales_prediction.py
```

## Outputs
- `outputs/actual_vs_predicted.png`
- `outputs/predictions.csv`
- `outputs/metrics.txt`

## Evaluation
The script reports MAE, RMSE, and R² on a held-out test set. Results depend on the downloaded dataset and split; do not copy sample metrics without running the code.
