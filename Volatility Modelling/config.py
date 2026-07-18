"""Configuration for Project 8 — Volatility Forecasting.

This module provides a ``settings`` object with project-wide
configuration values used by ``model.py`` and ``main.py``.

Attributes
----------
settings : Settings
    Singleton configuration object.

    Attributes
    ----------
    db_name : str
        SQLite database filename (default
        ``"stocks.sqlite"``).
    model_directory : str
        Directory for saved model files (default
        ``"models"``).
"""

# REMOVEBLOCK START
from __future__ import annotations


class Settings:
    """Project-wide configuration container.

    Attributes
    ----------
    db_name : str
        SQLite database filename.
    model_directory : str
        Directory for saved model ``.pkl`` files.
    """

    db_name: str = "stocks.sqlite"
    model_directory: str = "models"


settings = Settings()
# REMOVEBLOCK END
