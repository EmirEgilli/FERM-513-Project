# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:39:42 2023

@author: emir.e
"""

# Let's start with importing the necessary libraries as we go
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def banks_norm(ticker, start = "2007-01-01", end = "2023-11-01"):
    # Download stock data
    stocks = yf.download(ticker, start, end, interval="1mo")
    stocks.to_csv("bank_stocks.csv")

    # Read data and process
    stocks = pd.read_csv("bank_stocks.csv", header=[0, 1], index_col=[0], parse_dates=[0])
    stocks.columns = stocks.columns.to_flat_index()
    stocks.columns = pd.MultiIndex.from_tuples(stocks.columns)
    stocks.swaplevel(axis=1).sort_index(axis=1)
    close = stocks.loc[:, "Adj Close"].copy()

    # Calculate normalized returns
    norm = close.div(close.iloc[0]).mul(1)
    norm["BANK_N"] = norm.mean(axis=1)

    # Add "NORM" column to the DataFrame
    stocks["BANK_N"] = norm["BANK_N"]

    # Save DataFrame to CSV file with the added "NORM" column
    stocks.to_csv("bank_stocks.csv")

    # Plot the data
    plt.figure(figsize=(10, 6))
    for column in norm.columns[:-1]:  # Exclude the "NORM" column
        plt.plot(norm.index, norm[column], label=column, linewidth=2)  # Set linewidth for individual lines

    # Plot the "Mean" line with a wider linewidth
    plt.plot(norm.index, norm["BANK_N"], label="BANK_N", linewidth=4, color='red')

    plt.legend(fontsize=13)
    plt.title("Fig. 1 - Normalized Returns", fontsize=20)
    plt.grid(True)
    plt.show()
    
banks_norm(["YKBNK.IS", "ISCTR.IS", "AKBNK.IS", "GARAN.IS", "HALKB.IS", "VAKBN.IS"], "2007-01-01", "2023-11-01")

# We need to add the additional library statsmodels at this point
import statsmodels.formula.api as smf

reg_data = pd.read_csv("bank_stocks_norm2.csv", index_col='Date')

# 𝑅𝑏𝑎𝑛𝑘=𝛽1𝐶𝑃𝐼𝑦𝑦+𝛽2𝑑+𝛽3𝑟+𝜖
model1 = smf.ols('XBANK ~ CPI + DEP_BAL + IR', data = reg_data).fit()
print(model1.summary())

# 𝑅𝑏𝑎𝑛𝑘=𝛽1𝐶𝑃𝐼𝑦𝑦+𝛽2𝑑+𝛽3𝑟+𝛽4𝑟𝐷1+𝜖
model2 = smf.ols('XBANK ~ CPI + DEP_BAL + IR * Dummy_d', data = reg_data).fit()
print(model2.summary())

# Read data from CSV file
df = pd.read_csv('bank_stocks_norm2.csv')

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' column as the index
df.set_index('Date', inplace=True)

# Create subplots with 2 rows and 2 columns
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# Plot XBANK
axes[0, 0].plot(df['XBANK'], linestyle='-')
axes[0, 0].set_title('XBANK')
axes[0, 0].set_xlabel('Year')
axes[0, 0].grid(True)

# Plot Consumer Price Index
axes[0, 1].plot(df['CPI'], linestyle='-')
axes[0, 1].set_title('Consumer Price Index')
axes[0, 1].set_xlabel('Year')
axes[0, 1].grid(True)

# Plot Deposit Balance
axes[1, 0].plot(df['DEP_BAL'], linestyle='-')
axes[1, 0].set_title('Deposit Balance')
axes[1, 0].set_xlabel('Year')
axes[1, 0].grid(True)

# Plot Interest Rates
axes[1, 1].plot(df['IR'], linestyle='-')
axes[1, 1].set_title('Interest Rates')
axes[1, 1].set_xlabel('Year')
axes[1, 1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

