# Proposal 1 — Student Service Request Intelligence

## Run locally
1. Install Python 3.10 or newer.
2. Extract this folder.
3. Open Command Prompt inside this folder.
4. Run `python -m pip install -r requirements.txt`.
5. Run `python -m streamlit run app.py`.

Or double-click `RUN_APP.bat` on Windows.

## Included files
- `app.py` — Streamlit dashboard and model evaluation.
- `data/01_student_service_requests_synthetic.csv` — sample dataset.
- `requirements.txt` — Python dependencies.
- `.streamlit/config.toml` — theme settings.

## Main features
- KPI cards and filters for office, request type, and date.
- Request volume, status, monthly trend, and processing-time analysis.
- Decision Tree and Random Forest comparison using a held-out split.
- Accuracy, precision, recall, F1, and confusion matrix.
- CSV download of filtered records.

## Important methodology note
The model uses only Request_Type, Office, Priority, and Service_Target_Days. `Date_Completed` and `Processing_Days` are deliberately excluded from predictors to prevent target leakage. Only completed records with known Yes/No delay labels are used for training/evaluation.

The included dataset is synthetic and must not be reported as real school data or real model findings. Obtain permission and use anonymized school records for actual research.
