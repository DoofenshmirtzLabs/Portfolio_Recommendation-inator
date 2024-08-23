import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
import os
import openai

# Initialize the OpenAI client with your API key
client = openai.OpenAI(api_key='')

# Function to build the autoencoder model
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

# Function to train the autoencoder and sort stocks by RMSE
def train_autoencoder_and_sort_rmse(df,len_df,epochs=50, batch_size=32):
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Build and train the autoencoder
    input_dim = scaled_data.shape[1]
    autoencoder = build_autoencoder(input_dim)
    
    # Train the model
    autoencoder.fit(scaled_data, scaled_data, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    
    # Predict (reconstruct) the data
    reconstructed_data = autoencoder.predict(scaled_data)
    
    # Calculate RMSE for each stock
    rmse_list = []
    for i in range(scaled_data.shape[1]):
        rmse = np.sqrt(mean_squared_error(scaled_data[:, i], reconstructed_data[:, i]))
        rmse_list.append((df.columns[i], rmse))
    
    # Sort stocks by RMSE
    rmse_list_sorted = sorted(rmse_list, key=lambda x: x[1])
    
    # Print the stocks ordered by least to highest RMSE
    
   
    k = int(len_df * 0.20)  # This will truncate any decimal

    rmse_high=rmse_list_sorted[:k+1]
    rmse_low=rmse_list_sorted[-k:]
    return rmse_high,rmse_low


def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path)
    df = df.rename(columns={'Unnamed: 0': 'date'})
    newdf = df.drop('date', axis=1)
    return newdf

def calculate_annualized_returns(newdf, years=5):
    w_df = newdf
    w_df_subtracted = w_df - w_df.iloc[0]
    df_subtracted = w_df_subtracted.iloc[1:]
    percentage_returns = df_subtracted.iloc[-1] / newdf.iloc[0]
    annualized_returns = percentage_returns * 100 * years
    return annualized_returns.to_dict()

def get_user_input():
    initial_investment = float(input("Please enter your initial investment amount: "))
    expected_returns = float(input("Please enter your expected amount to be made: "))
    risk_tolerance = input("Please enter your risk tolerance (low, medium, high): ")
    time_period = int(input("Please enter the time period of your investment (in years): "))
    return initial_investment, expected_returns, risk_tolerance, time_period

def calculate_daily_returns(df):
    returns = df.pct_change().dropna()
    return returns

def calculate_average_annual_return(returns, num_trading_days_per_year=252):
    average_daily_return = returns.mean()
    average_annual_return_percentage = average_daily_return * 100 * num_trading_days_per_year
    return average_annual_return_percentage

def calculate_variance(returns):
    variance = returns.var() * 5
    return variance.to_dict()

def calculate_sharpe_ratio(returns, rf=0.02):
    excess_returns = returns.mean() - rf
    std_dev = returns.std()
    sharpe_ratio = excess_returns / std_dev
    return sharpe_ratio

def classify_customer(expected_returns, initial_investment, time_period, annualized_returns):
    avgreturns = (expected_returns / initial_investment) ** (1 / time_period) - 1
    avgreturnspercentage = avgreturns * 100
    percentiles = np.percentile(list(annualized_returns.values()), [25, 50, 75])
    
    if avgreturnspercentage < percentiles[0]:
        customer_expected_return = 'Low'
    elif avgreturnspercentage > percentiles[2]:
        customer_expected_return = 'High'
    else:
        customer_expected_return = 'Medium'
    
    return customer_expected_return, avgreturnspercentage

def classify_stocks(annualized_returns, variance):
    annualized_returns_values = list(annualized_returns.values())
    returns_percentiles = np.percentile(annualized_returns_values, [25, 50, 75])
    variance_values = list(variance.values())
    variances_percentiles = np.percentile(variance_values, [25, 50, 75])
    
    classified_stocks = {}
    for stock, returns in annualized_returns.items():
        variance_value = variance[stock]
        returns_classification = (
            'Low' if returns < returns_percentiles[0] else
            'High' if returns > returns_percentiles[2] else
            'Medium'
        )
        risk_classification = (
            'Low' if variance_value < variances_percentiles[0] else
            'High' if variance_value > variances_percentiles[2] else
            'Medium'
        )
        classified_stocks[stock] = {'returns': returns_classification, 'risk': risk_classification}
    
    return classified_stocks

def matching_stocks(classified_stocks, customer_classification, risk_tolerance):
    matching_stocks = [
        stock for stock, classification in classified_stocks.items()
        if classification['returns'] == customer_classification and classification['risk'] == risk_tolerance
    ]
    not_matching_stocks = [
        stock for stock in classified_stocks if stock not in matching_stocks
    ]
    return matching_stocks, not_matching_stocks

def generate_random_portfolio(matching_stocks, other_stocks, min_portfolio_size, max_portfolio_size):
    portfolio_size = random.randint(min_portfolio_size, max_portfolio_size)
    
    if portfolio_size < len(matching_stocks):
        portfolio = random.sample(matching_stocks, portfolio_size)
    else:
        portfolio = matching_stocks.copy()
        num_other_stocks = portfolio_size - len(matching_stocks)
        if num_other_stocks > 0:
            portfolio.extend(random.sample(other_stocks, min(num_other_stocks, len(other_stocks))))
    
    # Return both portfolio and stock names
    stock_names = portfolio  # If you have a specific way to determine stock names, adjust accordingly
    return portfolio, stock_names


def generate_random_portfolios(matching, not_matching, num_portfolios=100, min_portfolio_size=1, max_portfolio_size=30):
    random_portfolios = [
        generate_random_portfolio(matching, not_matching, min_portfolio_size, max_portfolio_size)
        for _ in range(num_portfolios)
    ]
    return random_portfolios

def calculate_portfolio_metrics(newdf, random_portfolios, num_portfolios=1000):
    all_portfolios = []
    all_returns = []
    all_risks = []
    all_sharpe_ratios = []

    expected_returns_global = {}
    expected_risk_global = {}
    expected_sharpe_ratio_global = {}
    optimal_portfolio_weights = {}
    
    optimal_portfolio = None
    optimal_stock_names = []

    for portfolio, stock_names in random_portfolios:
        portfolio_tuple = tuple(portfolio)
        portfolio_len = len(portfolio)
        
        expected_returns_local = np.zeros(num_portfolios)
        expected_risk_local = np.zeros(num_portfolios)
        expected_sharpe_ratio_local = np.zeros(num_portfolios)
        weights_local = []
        
        concatenated_data = pd.concat([newdf[stock] for stock in portfolio], axis=1)
        
        # Filter out any rows with zero or negative values to avoid log issues
        concatenated_data = concatenated_data[(concatenated_data > 0).all(axis=1)].dropna()
        
        # Calculate logarithmic returns
        log_returns_local = np.log(concatenated_data / concatenated_data.shift(1)).dropna()
        
        for k in range(num_portfolios):
            w = np.random.random(portfolio_len)
            w /= np.sum(w)
            
            mean_log_return = log_returns_local.mean()
            sigma = log_returns_local.cov()
            
            expected_returns_local[k] = np.sum(mean_log_return * w)
            expected_risk_local[k] = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
            expected_sharpe_ratio_local[k] = expected_returns_local[k] / expected_risk_local[k]
            
            weights_local.append(w)
        
        all_portfolios.extend([portfolio_tuple] * num_portfolios)
        all_returns.extend(expected_returns_local)
        all_risks.extend(expected_risk_local)
        all_sharpe_ratios.extend(expected_sharpe_ratio_local)

        max_index_local = expected_sharpe_ratio_local.argmax()
        max_sharpe_ratio = expected_sharpe_ratio_local[max_index_local]
        
        expected_returns_global[portfolio_tuple] = expected_returns_local[max_index_local]
        expected_risk_global[portfolio_tuple] = expected_risk_local[max_index_local]
        expected_sharpe_ratio_global[portfolio_tuple] = max_sharpe_ratio
        optimal_portfolio_weights[portfolio_tuple] = weights_local[max_index_local]
        
        # Save the stock names and portfolio with the highest Sharpe Ratio
        if optimal_portfolio is None or expected_sharpe_ratio_global[portfolio_tuple] > max_sharpe_ratio:
            optimal_portfolio = portfolio_tuple
            optimal_stock_names = stock_names

    return expected_returns_global, expected_risk_global, expected_sharpe_ratio_global, optimal_portfolio_weights, all_returns, all_risks, all_sharpe_ratios, optimal_portfolio, optimal_stock_names

def plot_portfolios(all_returns, all_risks, all_sharpe_ratios, expected_returns_global, expected_risk_global, expected_sharpe_ratio_global):
    # Find the optimal portfolio (with the highest Sharpe Ratio)
    max_portfolio = max(expected_sharpe_ratio_global, key=expected_sharpe_ratio_global.get)
    
    # Plot all portfolios with scatter plot (color-coded by Sharpe Ratio)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(all_risks, all_returns, c=all_sharpe_ratios, cmap='viridis', marker='o', s=30)
    
    # Highlight the optimal portfolio with an "X"
    plt.scatter(expected_risk_global[max_portfolio], expected_returns_global[max_portfolio], marker='X', c='red', s=200, label="Optimal Portfolio")
    
    # Add labels and color bar
    plt.xlabel('Expected Risk')
    plt.ylabel('Expected Returns')
    plt.title('Portfolios - Expected Returns vs. Risk')
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.legend()
    
    plt.show()
def create_matching_stock_df(df, matching):
    # Initialize an empty DataFrame
    new_df = pd.DataFrame()

    # Loop through each stock in the matching list
    for stock in matching:
        if stock in df.columns:
            # Concatenate stock data to the new DataFrame along axis 1 (columns)
            new_df = pd.concat([new_df, df[[stock]]], axis=1)

    return new_df
def print_optimal_portfolio(optimal_portfolio, optimal_portfolio_weights, initial_investment):
    print("\nOptimal Portfolio:")
    for stock, weight in zip(optimal_portfolio, optimal_portfolio_weights):
        investment_amount = initial_investment * weight
        print(f"Stock: {stock}, Weight: {weight:.2f}, Amount Invested: ${investment_amount:.2f}")

def display_pie_chart(optimal_portfolio_weights, stock_names, initial_investment):
    # Ensure optimal_portfolio_weights is a dictionary and stock_names is a list
    weights = optimal_portfolio_weights
    amounts_invested = {stock: initial_investment * weight for stock, weight in zip(stock_names, weights)}

    # Prepare data for the pie chart
    labels = list(amounts_invested.keys())
    sizes = list(amounts_invested.values())
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Investment Distribution in Optimal Portfolio')
    plt.show()

# Function to plot portfolio value over time
def plot_portfolios(all_returns, all_risks, all_sharpe_ratios, expected_returns_global, expected_risk_global, expected_sharpe_ratio_global):
    # Find the optimal portfolio (with the highest Sharpe Ratio)
    max_portfolio = max(expected_sharpe_ratio_global, key=expected_sharpe_ratio_global.get)
    
    # Plot all portfolios with scatter plot (color-coded by Sharpe Ratio)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(all_risks, all_returns, c=all_sharpe_ratios, cmap='viridis', marker='o', s=30)
    
    # Highlight the optimal portfolio with an "X"
    plt.scatter(expected_risk_global[max_portfolio], expected_returns_global[max_portfolio], marker='X', c='red', s=200, label="Optimal Portfolio")
    
    # Add labels and color bar
    plt.xlabel('Expected Risk')
    plt.ylabel('Expected Returns')
    plt.title('Portfolios - Expected Returns vs. Risk')
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.legend()
    
    plt.show()
def display_portfolio_value_over_time(stock_prices_df, portfolio_weights, stock_names, initial_investment):
    """
    Displays portfolio value over time.

    Parameters:
    stock_prices_df (pd.DataFrame): DataFrame containing historical stock prices with columns as stock tickers.
    portfolio_weights (list): List of portfolio weights corresponding to each stock.
    stock_names (list): List of stock names corresponding to the weights.
    initial_investment (float): Initial investment amount.
    """
    # Ensure portfolio_weights and stock_names are lists of the same length
    if len(portfolio_weights) != len(stock_names):
        raise ValueError("The length of portfolio_weights and stock_names must be the same.")
    
    # Convert portfolio_weights to a dictionary with stock names as keys
    portfolio_weights_dict = dict(zip(stock_names, portfolio_weights))
    
    # Filter columns to include only those present in the DataFrame
    valid_columns = [stock for stock in portfolio_weights_dict.keys() if stock in stock_prices_df.columns]
    stock_prices_df_filtered = stock_prices_df[valid_columns]
    
    if stock_prices_df_filtered.empty:
        raise ValueError("No valid stocks found in the DataFrame columns.")
    
    # Calculate portfolio value over time
    portfolio_value = (stock_prices_df_filtered * [portfolio_weights_dict[stock] for stock in stock_prices_df_filtered.columns]).sum(axis=1)
    portfolio_value = initial_investment * (portfolio_value / portfolio_value.iloc[0])  # Normalize to initial investment
    
    # Plot the portfolio value over time
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value.index, portfolio_value, label='Portfolio Value', color='blue')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.show()



def get_portfolio_summary(userinputs,portfolio_weights, sharpe_ratio):
  # Define the prompt
  prompt = (
      f"As a senior financial consultant, please provide an executive summary of the following investment portfolio:\n\n"
      f"user inputs:{userinputs}\n\n"
      f"Portfolio Weights: {portfolio_weights}\n\n"
      f"Sharpe Ratio: {sharpe_ratio}\n\n"
      
      f"Please include a brief analysis of the portfolio's expected performance, risk profile, and how it aligns with typical investment goals. "
     
  )

  # Make the API call
  response = client.chat.completions.create(
      model="gpt-3.5-turbo",  # or gpt-4-turbo or gpt-4
      messages=[
          {"role": "user", "content": prompt}
      ],
      max_tokens=150  # Adjust the token limit based on your needs
  )

  # Extract and return the generated summary (updated line)
  return response.choices[0].message.content.strip()
# Main execution starts here
# Main execution starts here
file_path = 'C:\\Users\\user\\Downloads\\dataset1.xlsx'
newdf = load_and_prepare_data(file_path)
annualized_returns = calculate_annualized_returns(newdf)
initial_investment, expected_returns, risk_tolerance, time_period = get_user_input()
returns = calculate_daily_returns(newdf)
average_annual_return_percentage = calculate_average_annual_return(returns)
variance = calculate_variance(returns)
sharpe_ratio = calculate_sharpe_ratio(returns)
customer_classification, avgreturnspercentage = classify_customer(expected_returns, initial_investment, time_period, annualized_returns)
classified_stocks = classify_stocks(annualized_returns, variance)
matching, not_matching = matching_stocks(classified_stocks, customer_classification, risk_tolerance)
print(matching)
len_df = len(matching)
if len_df:
    new_df = create_matching_stock_df(newdf, matching)
else:
    new_df = create_matching_stock_df(newdf, not_matching)
rmse_high, rmse_low = train_autoencoder_and_sort_rmse(new_df, len_df)
random_portfolios = generate_random_portfolios(matching, not_matching)
expected_returns_global, expected_risk_global, expected_sharpe_ratio_global, optimal_portfolio_weights, all_returns, all_risks, all_sharpe_ratios, optimal_portfolio, optimal_stock_names = calculate_portfolio_metrics(newdf, random_portfolios)
plot_portfolios(all_returns, all_risks, all_sharpe_ratios, expected_returns_global, expected_risk_global, expected_sharpe_ratio_global)

# Display pie chart for the optimal portfolio
display_pie_chart(optimal_portfolio_weights[optimal_portfolio], optimal_stock_names, initial_investment)

# Display portfolio value over time
optimal_df = create_matching_stock_df(newdf, optimal_stock_names)
display_portfolio_value_over_time(optimal_df, optimal_portfolio_weights[optimal_portfolio], optimal_stock_names, initial_investment)
userinputs = {
    'initial_investment_amount': initial_investment,
    'user_expected_returns': expected_returns,
    'user_risk_tolerance': risk_tolerance,
    'time_period': time_period
}
optimal_sharpe_ratio = expected_sharpe_ratio_global[optimal_portfolio]
summary = get_portfolio_summary(userinputs, optimal_portfolio_weights[optimal_portfolio], optimal_sharpe_ratio)
print(summary)


