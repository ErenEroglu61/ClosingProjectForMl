import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
DATASETS = {
    "math": {
        "label": "Math score test verisi",
        "model_path": BASE_DIR / "Models" / "closing_project.pkl",
        "data_path": BASE_DIR / "Test_Data" / "test_data.csv",
    },
    "reading": {
        "label": "Reading score test verisi",
        "model_path": BASE_DIR / "Models" / "closing_project_reading_score.pkl",
        "data_path": BASE_DIR / "Test_Data" / "test_data_reading_score.csv",
    },
    "writing": {
        "label": "Writing score test verisi",
        "model_path": BASE_DIR / "Models" / "closing_project_writing_score.pkl",
        "data_path": BASE_DIR / "Test_Data" / "test_data_writing_score.csv",
    },
}

loaded_models: Dict[str, Dict[str, Any]] = {}

for dataset_key, spec in DATASETS.items():
    with spec["model_path"].open("rb") as file:
        saved_data = pickle.load(file)
    loaded_models[dataset_key] = {
        "label": spec["label"],
        "model": saved_data["model"],
        "columns": saved_data.get("columns"),
        "encoders": saved_data.get("encoders", {}),
        "maps": saved_data.get("maps", []),
        "scaler": saved_data.get("scaler"),
    }


def _load_test_data(dataset_key: str) -> pd.DataFrame:
    if dataset_key not in loaded_models:
        raise HTTPException(status_code=404, detail="Bilinmeyen test veri seti")

    path = DATASETS[dataset_key]["data_path"]
    data = pd.read_csv(path, header=0)
    expected_columns = loaded_models[dataset_key].get("columns")
    if expected_columns is not None and len(expected_columns) == len(data.columns):
        data.columns = expected_columns
    return data


def _prepare_manual_sample(dataset_key: str, features: Dict[str, Any]) -> pd.DataFrame:
    if dataset_key not in loaded_models:
        raise HTTPException(status_code=404, detail="Bilinmeyen test veri seti")

    dataset = loaded_models[dataset_key]
    expected_columns = dataset["columns"]
    if expected_columns is None:
        raise HTTPException(status_code=500, detail="Model için beklenen sütun bilgisi bulunamadı.")

    missing = [col for col in expected_columns if col not in features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Eksik özellikler: {', '.join(missing)}",
        )

    input_data = pd.DataFrame([{col: features[col] for col in expected_columns}])

    for col, encoder in dataset["encoders"].items():
        if col in input_data.columns:
            input_data[col] = encoder.transform(input_data[[col]])[:, 0]

    for index, col in enumerate(["lunch", "gender", "test preparation course"]):
        if col in input_data.columns:
            input_data[col] = input_data[col].map(dataset["maps"][index])

    return input_data


class PredictRequest(BaseModel):
    dataset: str
    row: Optional[int] = None
    features: Optional[Dict[str, Any]] = None


INDEX_PATH = BASE_DIR / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/preview")
async def preview(dataset: str = Query("math", description="Seçilecek test veri seti")):
    if dataset not in loaded_models:
        raise HTTPException(status_code=404, detail="Bilinmeyen test veri seti")

    data = _load_test_data(dataset)
    return {
        "dataset": dataset,
        "label": loaded_models[dataset]["label"],
        "row_count": len(data),
        "preview": data.head(5).to_dict(orient="records"),
        "columns": data.columns.tolist(),
        "input_columns": loaded_models[dataset].get("columns", []),
    }


@app.post("/predict")
async def predict(request: PredictRequest):
    dataset_key = request.dataset.lower()
    if dataset_key not in loaded_models:
        raise HTTPException(status_code=404, detail="Bilinmeyen test veri seti")

    model_spec = loaded_models[dataset_key]
    model = model_spec["model"]

    if request.features is not None:
        input_data = _prepare_manual_sample(dataset_key, request.features)
        prediction = model.predict(model_spec["scaler"].transform(input_data))
        return {
            "dataset": dataset_key,
            "label": model_spec["label"],
            "input": request.features,
            "prediction": prediction.tolist(),
        }

    data = _load_test_data(dataset_key)
    row = request.row or 0
    if row < 0 or row >= len(data):
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz satır numarası. 0 ile {len(data) - 1} arasında bir sayı girin.",
        )

    sample = data.iloc[[row]]
    prediction = model.predict(sample)
    return {
        "dataset": dataset_key,
        "label": model_spec["label"],
        "selected_row": int(row),
        "prediction": prediction.tolist(),
    }
