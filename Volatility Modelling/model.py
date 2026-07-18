"""Volatility modelling layer for Project 8.

This module contains the ``GarchModel`` class, which
encapsulates the full lifecycle of a GARCH volatility
model: data wrangling, fitting, predicting, saving, and
loading.

Overview
--------
Students build this class incrementally in Lesson 4.
Each method is written and tested inside the notebook
first (using ``assert`` statements), then moved here.

The class depends on the data layer built in earlier
lessons:

- ``SQLRepository`` (from ``data.py``) — to read stock
  data from the SQLite database.
- ``AlphaVantageAPI`` (from ``data.py``) — optionally
  used to fetch fresh data from the API when
  ``use_new_data=True``.

The trained model is serialized to disk with ``joblib``
so that the ``/predict`` endpoint in ``main.py`` can
load it later without retraining.

Constants
---------
DB_NAME : str
    Default SQLite database filename
    (e.g., ``"stocks.sqlite"``).
MODEL_DIRECTORY : str
    Default directory where trained model ``.pkl`` files
    are saved (e.g., ``"models"``).

Classes
-------
GarchModel
    End-to-end GARCH volatility model with methods for
    wrangling, fitting, predicting, saving, and loading.

    Attributes
    ----------
    ticker : str
        The stock ticker symbol this model instance is
        associated with (e.g., ``"AMBUJACEM.BSE"``).
    repo : SQLRepository
        A ``SQLRepository`` instance used to read data
        from the SQLite database and (when fetching new
        data) to insert fresh records.
    use_new_data : bool
        If ``True``, the ``wrangle_data`` method fetches
        fresh data from the API (via ``AlphaVantageAPI``)
        and stores it in the database before computing
        returns. If ``False``, it reads existing data
        from the database.
    model_directory : str
        Path to the directory where trained model files
        are saved. Defaults to ``MODEL_DIRECTORY``.
    data : pd.Series or None
        The return series produced by ``wrangle_data``.
        ``None`` until ``wrangle_data`` is called.
    model : ARCHModelResult or None
        The fitted GARCH model object. ``None`` until
        ``fit`` or ``load`` is called.

    Methods
    -------
    wrangle_data(n_observations)
        Retrieve stock data through ``self.repo`` (and
        optionally fetch new data from the API if
        ``self.use_new_data`` is ``True``), compute daily
        percentage returns, keep the most recent
        ``n_observations`` values, and store the result
        in ``self.data``.

        Parameters
        ----------
        n_observations : int
            Number of return observations to keep.

        Returns
        -------
        None
            The result is stored in ``self.data``.

    fit(p, q)
        Build and fit a GARCH(p, q) model on
        ``self.data`` using ``arch.arch_model``. The
        fitted result is stored in ``self.model``.

        Parameters
        ----------
        p : int
            Number of lagged squared residuals (ARCH
            order).
        q : int
            Number of lagged variances (GARCH order).

        Returns
        -------
        None
            The result is stored in ``self.model``.

    predict_volatility(horizon)
        Use ``self.model`` to forecast volatility for
        the next ``horizon`` business days. Returns a
        dictionary mapping ISO 8601 date strings to
        predicted volatility values (standard deviation,
        not variance).

        Internally calls a private helper
        ``__clean_prediction`` that:

        1. Extracts the variance forecast from
           ``self.model.forecast(horizon=horizon)``.
        2. Generates a business-day date range starting
           the day after the last observation.
        3. Takes the square root of each variance to
           get volatility.
        4. Returns a ``dict[str, float]``.

        Parameters
        ----------
        horizon : int
            Number of business days to forecast.

        Returns
        -------
        dict
            Keys are ISO 8601 date strings.
            Values are predicted volatility (float).

    dump()
        Save ``self.model`` to disk using
        ``joblib.dump``. The filename includes a
        timestamp and the ticker symbol so each training
        run produces a unique file. The file is saved
        inside ``self.model_directory``.

        Returns
        -------
        str
            The full file path of the saved model.

    load()
        Find the most recent ``.pkl`` file matching
        ``self.ticker`` in ``self.model_directory``,
        load it with ``joblib.load``, and assign it to
        ``self.model``. Raises ``Exception`` if no
        matching file is found.

        Returns
        -------
        None
            The result is stored in ``self.model``.

Notes
-----
- The ``mock_alpha`` module must be activated before
  calling ``wrangle_data`` with ``use_new_data=True``.
  When ``use_new_data=False``, no API call is made —
  data is read directly from the SQLite database.

- Students should use IPython autoreload when working
  across lessons::

      %load_ext autoreload
      %autoreload 2
      from model import GarchModel

- The ``dump`` / ``load`` pattern uses ``glob`` to find
  the most recent file alphabetically. This works because
  filenames start with a timestamp
  (e.g., ``"2025-03-20T14:30:00_AMBUJACEM.BSE.pkl"``).

Examples
--------
>>> from model import GarchModel
>>> from data import SQLRepository
>>> import sqlite3
>>> conn = sqlite3.connect("stocks.sqlite")
>>> repo = SQLRepository(connection=conn)
>>> gm = GarchModel(
...     ticker="AMBUJACEM.BSE",
...     repo=repo,
...     use_new_data=False,
... )
>>> gm.wrangle_data(n_observations=2000)
>>> gm.fit(p=1, q=1)
>>> forecast = gm.predict_volatility(horizon=5)
>>> print(forecast)
{'2025-03-21T00:00:00': 1.234, ...}
>>> path = gm.dump()
>>> print(path)
models/2025-03-20T14:30:00_AMBUJACEM.BSE.pkl
"""

# REMOVEBLOCK START
from __future__ import annotations

import os
from datetime import datetime
from glob import glob

import joblib
import numpy as np
import pandas as pd
from arch import arch_model

from config import settings
from data import AlphaVantageAPI, SQLRepository

DB_NAME: str = settings.db_name
MODEL_DIRECTORY: str = settings.model_directory


class GarchModel:
    """End-to-end GARCH volatility model.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    repo : SQLRepository
        Database repository for reading/writing data.
    use_new_data : bool
        If ``True``, fetch fresh data from the API.
    """

    def __init__(
        self,
        ticker: str,
        repo: SQLRepository,
        use_new_data: bool,
    ) -> None:
        self.ticker = ticker
        self.repo = repo
        self.use_new_data = use_new_data
        self.model_directory = MODEL_DIRECTORY
        self.data: pd.Series | None = None
        self.model = None

    def wrangle_data(self, n_observations: int) -> None:
        """Retrieve data and compute returns.

        Parameters
        ----------
        n_observations : int
            Number of return observations to keep.
        """
        if self.use_new_data:
            av = AlphaVantageAPI()
            new_data = av.get_daily(ticker=self.ticker)
            self.repo.insert_table(
                table_name=self.ticker,
                records=new_data,
                if_exists="replace",
            )

        df = self.repo.read_table(table_name=self.ticker)
        df.sort_index(inplace=True)
        df["return"] = df["close"].pct_change() * 100
        self.data = df["return"].dropna().tail(n_observations)

    def fit(self, p: int, q: int) -> None:
        """Fit a GARCH(p, q) model on ``self.data``.

        Parameters
        ----------
        p : int
            ARCH order.
        q : int
            GARCH order.
        """
        model = arch_model(self.data, p=p, q=q, rescale=False)
        self.model = model.fit(disp="off")

    def __clean_prediction(self, horizon: int) -> dict[str, float]:
        """Convert raw forecast to dict.

        Parameters
        ----------
        horizon : int
            Forecast horizon in business days.

        Returns
        -------
        dict[str, float]
            ISO 8601 date strings to volatility.
        """
        forecast = self.model.forecast(horizon=horizon)
        variance = forecast.variance.iloc[-1]

        last_date = self.model.resid.index[-1]
        dates = pd.bdate_range(
            start=last_date + pd.DateOffset(days=1),
            periods=horizon,
        )

        prediction = {}
        for i, date in enumerate(dates):
            prediction[date.isoformat()] = float(np.sqrt(variance.iloc[i]))
        return prediction

    def predict_volatility(self, horizon: int) -> dict[str, float]:
        """Forecast volatility for *horizon* days.

        Parameters
        ----------
        horizon : int
            Number of business days to forecast.

        Returns
        -------
        dict[str, float]
            ISO 8601 date keys, volatility values.
        """
        return self.__clean_prediction(horizon)

    def dump(self) -> str:
        """Save ``self.model`` to disk.

        Returns
        -------
        str
            File path of the saved model.
        """
        os.makedirs(self.model_directory, exist_ok=True)
        timestamp = datetime.now().isoformat()
        filename = os.path.join(
            self.model_directory,
            f"{timestamp}_{self.ticker}.pkl",
        )
        joblib.dump(self.model, filename)
        return filename

    def load(self) -> None:
        """Load the most recent model from disk.

        Raises
        ------
        Exception
            If no saved model is found for the ticker.
        """
        pattern = os.path.join(
            self.model_directory,
            f"*{self.ticker}.pkl",
        )
        files = sorted(glob(pattern))
        if not files:
            raise Exception(f"No model found for {self.ticker}")
        self.model = joblib.load(files[-1])


# REMOVEBLOCK END
