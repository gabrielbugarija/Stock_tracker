from datetime import date, datetime
import pandas as pd
import yfinance as yfs
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import time

# Get user input for stock name and number of years
whichstock = input("Which stock? > ")
howmanyyears = int(input("How many years? > "))

# Download stock data from Yahoo Finance
today = datetime.today()
END_DATE = today.strftime('%Y-%m-%d')
START_DATE = today.replace(year=today.year - howmanyyears).strftime('%Y-%m-%d')
data = yfs.download(whichstock, start=START_DATE, end=END_DATE)

# Reset index and add 'Date' feature
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data.Date)

# Calculate Exponential Moving Average for 50 and 200 days
data['EMA-50'] = data['Close'].ewm(span=50, adjust=False).mean()
data['EMA-200'] = data['Close'].ewm(span=200, adjust=False).mean()

# Plot High vs Low prices
plt.figure(figsize=(8, 4))
plt.plot(data['Date'], data['Low'], label="Low", color="indianred")
plt.plot(data['Date'], data['High'], label="High", color="mediumseagreen")
plt.ylabel('Price (in USD)')
plt.xlabel("Time")
plt.title(f"High vs Low of {whichstock}")
plt.tight_layout()
plt.legend()
plt.show()

# Plot Exponential Moving Average
plt.figure(figsize=(8, 4))
plt.plot(data['Date'], data['EMA-50'], label="EMA for 50 days")
plt.plot(data['Date'], data['EMA-200'], label="EMA for 200 days")
plt.plot(data['Date'], data['Adj Close'], label="Close")
plt.title(f'Exponential Moving Average for {whichstock}')
plt.ylabel('Price (in USD)')
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.show()

# Prepare data for Polynomial Regression model
x = data[['Open', 'High', 'Low', 'Volume', 'EMA-50', 'EMA-200']]
y = data['Close']

# Create a PolynomialFeatures object with degree 2 (for quadratic polynomial)
poly_features = PolynomialFeatures(degree=2)

# Transform the features into polynomial features
x_poly = poly_features.fit_transform(x)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x_poly, y, test_size=0.2, random_state=42)

# Create and fit Polynomial Regression model
poly_model = LinearRegression()
poly_model.fit(X_train, y_train)

# Make predictions
pred = poly_model.predict(X_test)

# Plot Real vs Predicted values
plt.figure(figsize=(8, 4))
plt.plot(y_test, label="Actual Price")
plt.plot(pred, label="Predicted Price")
plt.xlabel("Data Points")
plt.ylabel("Price")
plt.title(f"Real vs Predicted Prices for {whichstock}")
plt.legend()
plt.tight_layout()
plt.show()

# Print Real vs Predicted prices
d = pd.DataFrame({'Actual_Price': y_test, 'Predicted_Price': pred})
print(d.head(10))
print(d.describe())