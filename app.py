import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import gradio as gr

# Function to fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data fetched.")
        return data['Adj Close']
    except Exception as e:
        return f"Error fetching data for {ticker}: {e}"

# Function to calculate log returns
def calculate_log_returns(prices):
    daily_returns = prices.pct_change()
    return np.log(1 + daily_returns)

# Function to perform Monte Carlo simulation
def monte_carlo_simulation(log_returns, last_price, days, iterations):
    u = log_returns.mean()
    v = log_returns.var()
    drift = u - (0.5 * v)
    stddev = log_returns.std()

    daily_return_paths = np.exp(drift + stddev * norm.ppf(np.random.rand(days, iterations)))
    price_list = np.zeros((days, iterations))
    price_list[0] = last_price

    for i in range(1, days):
        price_list[i] = price_list[i-1] * daily_return_paths[i]

    return price_list

# Function to analyze the results
def analyze_results(price_list, simulation_days):
    mean_final_price = np.mean(price_list[-1])
    percentiles = np.percentile(price_list[-1], [5, 25, 50, 75, 95])

    result_text = (
        f"Mean final price: ${mean_final_price:.2f}\n"
        f"Possible price range (5th - 95th percentile): ${percentiles[0]:.2f} - ${percentiles[4]:.2f}\n"
        f"The price has a 95% chance of not exceeding: ${percentiles[4]:.2f}"
    )

    return result_text

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 📈Stock Price Forecasting-Monte Carlo Simulation⚡")

    ticker = gr.Textbox(label="Stock Ticker Symbol(Search web if you don't know for a stock) ", placeholder="Enter stock ticker (e.g., AAPL)")
    start_date = gr.Textbox(label="Start Date(Recommendation: at least 4-5 years earlier)", placeholder="YYYY-MM-DD")
    prediction_date = gr.Textbox(label="Prediction Date", placeholder="YYYY-MM-DD")
    simulation_iterations = gr.Textbox(label="Number of Simulations(Recommendation: at least 100,000)", placeholder="Enter 'default' for 100,000", value="100000")

    result_text = gr.Textbox(label="Simulation Results(In Dollars)")

    submit_button = gr.Button("Run Simulation")
    submit_button.click(fn=lambda ticker, start_date, prediction_date, simulation_iterations: 
                        analyze_results(
                            monte_carlo_simulation(
                                calculate_log_returns(
                                    fetch_stock_data(ticker, start_date, '2024-01-01')
                                ), 
                                fetch_stock_data(ticker, start_date, '2024-01-01').iloc[-1], 
                                (pd.to_datetime(prediction_date) - fetch_stock_data(ticker, start_date, '2024-01-01').index[-1]).days, 
                                int(simulation_iterations)
                            ), 
                            (pd.to_datetime(prediction_date) - fetch_stock_data(ticker, start_date, '2024-01-01').index[-1]).days
                        ) if isinstance(fetch_stock_data(ticker, start_date, '2024-01-01'), pd.Series) else 
                        fetch_stock_data(ticker, start_date, '2024-01-01'), 
                        inputs=[ticker, start_date, prediction_date, simulation_iterations], 
                        outputs=result_text)

# Launch the Gradio app
demo.launch()
