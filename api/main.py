import pickle
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from xgboost import XGBRegressor


class Regressor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self) -> XGBRegressor:
        # This could be easily replaced by a File Storage System integration. (GCS or S3)
        with open(self.model_path, 'rb') as file:
            model = pickle.load(file)

        return model

    def _pre_process(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(lambda x: x.fillna(x.mean()))

        none_as_category_cols = ['Mas Vnr Type', 'Alley', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1',
                                 'BsmtFin Type 2',
                                 'Fireplace Qu', 'Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond',
                                 'Pool QC',
                                 'Fence',
                                 'Misc Feature']
        for column in none_as_category_cols:
            df[column] = df[column].fillna('NA')
            df[column] = df[column].astype('category')

        fill_as_zero_cols = ['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', 'Bsmt Full Bath',
                             'Garage Yr Blt', 'Bsmt Half Bath', 'Garage Cars', 'Garage Area']
        for column in fill_as_zero_cols:
            df[column] = df[column].fillna(0)

        return df

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._pre_process(data))


app = FastAPI(title="XGBoost Regression API")
model = Regressor("model/xgb_model.pkl")


class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="List of data points as dicts")


@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        df = pd.DataFrame(request.data)

        if df.empty:
            raise ValueError("Input data is empty.")

        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
