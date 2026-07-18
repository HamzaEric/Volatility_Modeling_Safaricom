import gradio as gr
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
from arch import arch_model

# --- Configuration & Initialization ---
MODEL_PATH = 'GARCH_Model/garch_production_model.pkl'
DATA_PATH = 'Datafiles/percentage returns.csv'
NSE_RED = '#46b965'

print("Loading model and history...")
try:
    # Load the serialized production model (trained on % returns)
    loaded_model_result = joblib.load(MODEL_PATH)
    # Load historical data (index_col=0 ensures Date is the index)
    historical_data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print("Files loaded successfully.")
except FileNotFoundError:
    print(f"Error: Could not find {MODEL_PATH} or {DATA_PATH}. Run Step 1 first.")
    exit()


# --- The Core Forecast Function ---
def generate_safaricom_forecast(horizon, yesterday_price, today_price):
    """
    Inputs raw prices, calculates daily percentage return, updates model history,
    and returns tomorrow's summary text and a forecast chart.
    """

    # 1. Error Handling for Inputs
    if yesterday_price <= 0 or today_price <= 0:
        return "⚠️ Please enter valid, positive closing prices.", None, None

    # 2. Calculate today's percentage return (Match training scale)
    today_pct_return = ((today_price - yesterday_price) / yesterday_price) * 100
    return_confirmation = f"Today's calculated percentage return for Safaricom: **{today_pct_return:.2f}%**"

    # 3. Inject new return into historical series to create a live starting point
    # arch_model requires the full series to generate accurate conditional volatility starting states.
    latest_returns = historical_data['percentage returns'].copy()

    # Create a single row DataFrame for today using the current date
    now_date = pd.Timestamp(datetime.date.today())
    # Ensure this index is a DateTimeIndex to match historical_data
    today_series = pd.Series([today_pct_return], index=[now_date])

    # Concatenate history with today's new observation
    updated_returns_series = pd.concat([latest_returns, today_series])
    updated_returns_series.index.name = 'Date'

    # 4. Generate Multi-step-ahead Forecast (Dynamic)
    # Note: Using the 'data' parameter on a fitted result allows for out-of-sample
    # forecasting without needing the source data stored in the pickle.
    dynamic_model = arch_model(updated_returns_series, p=1, q=1, rescale=False)

    # Lock in the trained parameters (omega, alpha, beta) from the loaded model
    fixed_result = dynamic_model.fix(loaded_model_result.params)

    # Now generate the forecast using the fixed model (do not pass 'data' here)
    forecast_result = fixed_result.forecast(
        horizon=int(horizon),
        reindex=False
    )

    # Extract variance path (last row contains the multi-step horizon forecasts)
    variance_path = forecast_result.variance.iloc[-1, :].values

    # 5. Extract Standard Deviation (Volatility) and summary
    volatility_path = np.sqrt(variance_path)

    # Extract long-run volatility for comparison (sqrt(omega / (1 - alpha - beta)))
    # Accessing parameters from the model result object.
    params = loaded_model_result.params
    # Safe guard for potential mean models, ensuring index 0 is omega.
    omega_idx = params.index.get_loc('omega') if 'omega' in params.index else 0
    alpha_idx = params.index.get_loc('alpha[1]') if 'alpha[1]' in params.index else 1
    beta_idx = params.index.get_loc('beta[1]') if 'beta[1]' in params.index else 2

    omega = params.iloc[omega_idx]
    alpha = params.iloc[alpha_idx]
    beta = params.iloc[beta_idx]
    long_run_vol = np.sqrt(omega / (1 - alpha - beta))

    tomorrow_summary = (
        f" ## Forecast for: **Tomorrow**\n\n"
        f"Predicted Safaricom Volatility: **{volatility_path[0]:.3f}%**\n"
        f"Historical Long-Run Average Volatility: **{long_run_vol:.3f}%**\n\n"
        f"Model Parameters Check: α+β = **{(alpha + beta):.3f}** (Mean reverting if < 1)."
    )

    # 6. Create Plotly Chart of Predicted Volatility Curve
    days_ahead = list(range(1, int(horizon) + 1))

    fig = go.Figure()

    # Add Predicted Volatility Path
    fig.add_trace(go.Scatter(
        x=days_ahead,
        y=volatility_path,
        mode='lines+markers',
        name='Predicted Volatility Path',
        line=dict(color=NSE_RED, width=3),
        marker=dict(size=8)
    ))

    # Add Long-Run Volatility Line for Context
    fig.add_hline(
        y=long_run_vol,
        line_dash="dash",
        line_color="black",
        annotation_text="Historical Baseline Risk",
        annotation_position="bottom right"
    )

    # Chart Layout Styling
    fig.update_layout(
        title={
            'text': f"Safaricom {horizon}-Day Dynamic Volatility Forecast",
            'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        xaxis_title="Days in the Future",
        yaxis_title="Predicted Daily Volatility (%)",
        xaxis=dict(tickmode='linear', dtick=1),  # Tick every day
        yaxis=dict(rangemode='tozero'),  # Ensure Y starts at 0
        template='plotly_white'
    )

    return return_confirmation, tomorrow_summary, fig


# --- Gradio Interface Definition ---
# Using Option 2 inputs mapped to the dynamic forecasting function.

with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown(
        """
        # Safaricom Stock Volatility Model (NSE)
        
        ---
        An interactive quantitative tracking tool utilizing a GARCH(1,1) model to forecast daily conditional volatility and market risk for Safaricom equities on the Nairobi Securities Exchange.
        """
    )

    with gr.Blocks() as demo:

        gr.Markdown("""
           ## Safaricom Quantitative Volatility Dashboard
           ---
        """)
        # Everything inside this Row will be arranged horizontally
        with gr.Row():
            image_one = gr.Image(value="Images/newplot2.png",label="Sampling Conditional Volatility")
            image_two = gr.Image(value="Images/newplot.png",label="Walk Forward Backtesting")

    gr.Markdown("""
    ---
    ## Enter Daily Market Data For Forecast
    
    """)
    with gr.Row():
        # --- Left Column: Inputs ---
        with gr.Column(scale=1):
            horizon_input = gr.Slider(
                minimum=1, maximum=21, step=1, value=10,
                label="Forecast Horizon (Trading Days Ahead)"
            )
            yesterday_input = gr.Number(label="Safaricom Close Price (Yesterday KES)", value=30.00)
            today_input = gr.Number(label="Safaricom Close Price (Today KES)",
                                    value=30.45)  # Example input resulting in a 1.5% return

            submit_btn = gr.Button("Generate Forecast", variant="primary")

            # --- Right Column: Text Summary Outputs ---
            with gr.Column(scale=2):
                return_calculation_output = gr.Markdown(label="Data Validation")
                tomorrow_summary_output = gr.Markdown(label="Tomorrow's Projection")

    # --- Bottom Row: The Main Output Chart ---
    with gr.Row():
        with gr.Column(scale=1):
            chart_output = gr.Plot(label="Volatility Forecast Path")

    # --- Interaction Logic ---
    submit_btn.click(
        fn=generate_safaricom_forecast,
        inputs=[horizon_input, yesterday_input, today_input],
        outputs=[return_calculation_output, tomorrow_summary_output, chart_output]
    )

# --- Launch the App ---
if __name__ == "__main__":
    # share=False ensures it runs only on your Linux environment/local machine.
    interface.launch(share=False)