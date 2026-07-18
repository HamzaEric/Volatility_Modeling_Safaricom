# Safaricom Volatility FModel : GARCH(1,1) 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Gradio](https://img.shields.io/badge/Gradio-App-orange)

A quantitative web application designed specifically to forecast daily market volatility for Safaricom stock. Built with a serialized GARCH(1,1) model and an interactive Gradio interface, this tool allows for dynamic risk projection based on Safaricom's historical market data and recent returns.

## Overview

Financial time series exhibit volatility clustering—periods of high variance followed by high variance, and low variance followed by low variance. This project implements a **Generalized Autoregressive Conditional Heteroskedasticity (GARCH)** model to capture these dynamics and forecast future risk for Safaricom. 

The underlying model was rigorously backtested using an expanding-window walk-forward validation loop to prevent data leakage, and the final production weights are serialized via `joblib` for rapid, lightweight inference.

### The Math: GARCH(1,1)
The model forecasts conditional variance ($\sigma_t^2$) using the following equation:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

Where:
*   $\omega$: Baseline variance (constant).
*   $\alpha$: Sensitivity to new market shocks (ARCH term).
*   $\beta$: Persistence of past volatility (GARCH term).

##  Features

*   **Pre-Trained Production Model:** Utilizes a `joblib`-serialized GARCH model trained exclusively on Safaricom data for instant inference.
*   **Interactive Gradio Interface:** A clean, component-based UI that removes the need for terminal-based execution.
*   **Dynamic Forecasting:** Input custom horizons (days to forecast) and the latest daily logarithmic returns to adjust the forecast to current market conditions.


##  Project Structure

```text
├── app.py                         # Main Gradio application script
├── safaricom_garch_model.pkl      # Serialized arch model weights
├── data/
│   └── safaricom_historical.csv   # Cleaned Safaricom dataset
├── notebooks/
│   └── garch_walkforward.ipynb    # Training, validation, and EDA
├── requirements.txt               # Python dependencies
└── README.md
