Stock Prediction Web App

Overview

A Flask-based web app for fetching, visualizing, and predicting stock prices using Yahoo Finance (yfinance), Prophet, and Plotly.

Features

Fetch historical stock data

Interactive candlestick charts

Stock price predictions (30 days)

Requirements

Python 3.8+

Install dependencies:

pip install flask yfinance pandas numpy prophet plotly

Usage

Run the app:

python back2.py

Open http://127.0.0.1:5000/

Enter a stock symbol, select historical data duration, and submit.

Structure

back2.py - Main app script

templates/ - HTML templates

static/ - Static assets

Troubleshooting

Ensure correct stock symbols.

Sufficient historical data is required for predictions.

License

Open-source project.

Credits

Built using yfinance, Prophet, Flask, and Plotly.

