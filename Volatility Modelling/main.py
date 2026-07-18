"""FastAPI application for the Volatility Forecasting API.

This module defines the REST API that exposes the GARCH
volatility model built in earlier lessons. It contains
Pydantic data classes for request/response validation and
two POST endpoints: ``/fit`` (train a model) and
``/predict`` (generate a volatility forecast).

Components
----------
FitIn : pydantic.BaseModel
    Request schema for the ``/fit`` endpoint.

    Attributes
    ----------
    ticker : str
        Stock ticker symbol.
    use_new_data : bool
        Whether to fetch fresh data from the API.
    n_observations : int
        Number of return observations to use.
    p : int
        ARCH order for the GARCH model.
    q : int
        GARCH order for the GARCH model.

FitOut : FitIn
    Response schema for the ``/fit`` endpoint.
    Inherits all fields from ``FitIn`` and adds:

    Attributes
    ----------
    success : bool
        Whether the model was trained successfully.
    message : str
        Confirmation or error message.

PredictIn : pydantic.BaseModel
    Request schema for the ``/predict`` endpoint.

    Attributes
    ----------
    ticker : str
        Stock ticker symbol.
    n_days : int
        Forecast horizon in business days.

PredictOut : PredictIn
    Response schema for the ``/predict`` endpoint.
    Inherits all fields from ``PredictIn`` and adds:

    Attributes
    ----------
    success : bool
        Whether the forecast was generated successfully.
    forecast : dict
        Mapping of ISO 8601 date strings to predicted
        volatility values.
    message : str
        Confirmation or error message.

build_model(ticker, use_new_data)
    Helper function that creates a ``SQLRepository`` and
    wraps it in a ``GarchModel`` instance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    use_new_data : bool
        Whether to fetch fresh data from the API.

    Returns
    -------
    GarchModel
        A ready-to-use model instance.

Notes
-----
Start the server from the terminal with::

    uvicorn main:app --reload --workers 1 \\
        --host localhost --port 8008

Then interact with the API using ``requests.post``
from Python or visit http://localhost:8008/docs for
the interactive Swagger documentation.
"""

# REMOVEBLOCK START
from __future__ import annotations

import sqlite3

from fastapi import FastAPI
from pydantic import BaseModel

from config import settings
from data import SQLRepository
from model import GarchModel


class FitIn(BaseModel):
    """Request schema for the ``/fit`` endpoint.

    Attributes
    ----------
    ticker : str
        Stock ticker symbol.
    use_new_data : bool
        Whether to fetch fresh data from the API.
    n_observations : int
        Number of return observations to use.
    p : int
        ARCH order.
    q : int
        GARCH order.
    """

    ticker: str
    use_new_data: bool
    n_observations: int
    p: int
    q: int


class FitOut(FitIn):
    """Response schema for the ``/fit`` endpoint.

    Attributes
    ----------
    success : bool
        Whether the model trained successfully.
    message : str
        Confirmation or error message.
    """

    success: bool
    message: str


class PredictIn(BaseModel):
    """Request schema for the ``/predict`` endpoint.

    Attributes
    ----------
    ticker : str
        Stock ticker symbol.
    n_days : int
        Forecast horizon in business days.
    """

    ticker: str
    n_days: int


class PredictOut(PredictIn):
    """Response schema for the ``/predict`` endpoint.

    Attributes
    ----------
    success : bool
        Whether the forecast was generated.
    forecast : dict
        Date strings to volatility values.
    message : str
        Confirmation or error message.
    """

    success: bool
    forecast: dict
    message: str


def build_model(ticker: str, use_new_data: bool) -> GarchModel:
    """Create a GarchModel with a database connection.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    use_new_data : bool
        Whether to fetch fresh data from the API.

    Returns
    -------
    GarchModel
        A ready-to-use model instance.
    """
    connection = sqlite3.connect(settings.db_name, check_same_thread=False)
    repo = SQLRepository(connection=connection)
    return GarchModel(
        ticker=ticker,
        repo=repo,
        use_new_data=use_new_data,
    )


app = FastAPI()


@app.get("/hello")
def hello() -> dict[str, str]:
    """Health-check endpoint.

    Returns
    -------
    dict[str, str]
        A greeting message.
    """
    return {"message": "Hello from the volatility API!"}


@app.post("/fit", response_model=FitOut)
def fit_model(request: FitIn) -> FitOut:
    """Train a GARCH model and save it to disk.

    Parameters
    ----------
    request : FitIn
        Training parameters.

    Returns
    -------
    FitOut
        Training result with success status.
    """
    response = request.model_dump()
    try:
        model = build_model(
            ticker=request.ticker,
            use_new_data=request.use_new_data,
        )
        model.wrangle_data(n_observations=request.n_observations)
        model.fit(p=request.p, q=request.q)
        filename = model.dump()
        response["success"] = True
        response["message"] = f"Trained and saved: {filename}"
    except Exception as e:
        response["success"] = False
        response["message"] = str(e)
    return FitOut(**response)


@app.post("/predict", response_model=PredictOut)
def predict(request: PredictIn) -> PredictOut:
    """Generate a volatility forecast.

    Parameters
    ----------
    request : PredictIn
        Prediction parameters.

    Returns
    -------
    PredictOut
        Forecast result with predicted volatilities.
    """
    response = request.model_dump()
    try:
        model = build_model(
            ticker=request.ticker,
            use_new_data=False,
        )
        model.load()
        forecast = model.predict_volatility(horizon=request.n_days)
        response["success"] = True
        response["forecast"] = forecast
        response["message"] = "Forecast generated."
    except Exception as e:
        response["success"] = False
        response["forecast"] = {}
        response["message"] = str(e)
    return PredictOut(**response)


# REMOVEBLOCK END
