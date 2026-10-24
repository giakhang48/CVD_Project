# CVD_Project — Heart Disease Prediction

Machine learning project predicting cardiovascular disease risk from the `CVD_cleaned` dataset (BRFSS-derived, 308,854 records, available on Kaggle). Built for the DAP391m course at FPT University, Can Tho.

**Live demo:** [cardiovascular-disease-risk.streamlit.app](https://cardiovascular-disease-risk.streamlit.app/) *(Educational demo only — please do not enter real personal health information.)*

## What's in this repo

| File / Folder | Description |
|---|---|
| `CVD_analysis.ipynb` | Full analysis notebook (13 steps): encoding, feature engineering, EDA, advanced visualizations (Waffle, Area, Pie, Scatter, Word Cloud, Choropleth), class-imbalance check, balanced-strategy benchmark (4 models x 3 seeds), accuracy-priority benchmark without imbalance correction (9 models x 3 seeds), confusion matrix / ROC-AUC, feature importance, model export. |
| `CVD_cleaned.csv` | Cleaned source dataset (308,854 rows) used by the notebook. |
| `CVD_encoded.csv` | Encoded dataset with the three engineered features (`BMI_Category`, `Comorbidity_Count`, `Unhealthy_Diet_Score`). |
| `app.py` | Streamlit web app: heart disease risk prediction form + a TF-IDF/cosine-similarity chatbot answering questions about risk factors and the model. |
| `requirements.txt` | Python dependencies for the app. |
| `xgb_heart_disease_model.pkl` | Trained XGBoost model (balanced strategy, `scale_pos_weight`). |
| `model_feature_columns.pkl` | Exact feature column order expected by the model. |

## Key result

XGBoost trained with `scale_pos_weight` (about 11.4) to correct for severe class imbalance (91.9% negative / 8.1% positive) achieves **ROC-AUC = 0.816** on a stratified 80/20 test split (seed 42). At the default 0.5 threshold, recall for the Heart Disease class is 69% (1,549 of 4,994 true cases missed), precision is 22%, and accuracy is 0.77.

Accuracy alone looks worse than an unbalanced model (~0.92 accuracy), but the unbalanced models score a much lower F1 on the positive class (about 0.05-0.12 versus about 0.33), so they miss most true positive cases. See Steps 7-10 of the notebook for the comparison of both strategies.

## Running the Streamlit app

```
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Running the notebook

Open `CVD_analysis.ipynb` in Jupyter or VS Code. `CVD_cleaned.csv` is included in this repo and must be in the same folder as the notebook. In addition to the packages in `requirements.txt`, the notebook needs:

```
python3 -m pip install lightgbm catboost matplotlib seaborn plotly pywaffle wordcloud
```

## Disclaimer

This model is a screening/educational demonstration, not a diagnostic tool. It should not be used as a substitute for professional medical evaluation.
