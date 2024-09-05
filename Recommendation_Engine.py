import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

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
    expected_returns = float(input("Please enter your expected amount to be made "))
    risk_tolerance = input("Please enter your risk tolerance (low, medium, high): ")
    time_period = int(input("Please enter the time period of your investment (in years): "))
    return initial_investment, expected_returns, risk_tolerance, time_period

def calculate_daily_returns(df):
    returns = df / df.shift(1)
    returns = returns.drop(returns.index[0])
    return returns

def calculate_average_annual_return(returns, num_trading_days_per_year=252):
    average_daily_return = returns.mean()
    average_annual_return_percentage = average_daily_return * 100 * num_trading_days_per_year
    return average_annual_return_percentage

def calculate_variance(returns):
    variance = returns.var() * 5
    return variance.to_dict()

def calculate_sharpe_ratio(returns, rf=0.02):
    excess_returns = returns - rf
    mean_excess_return = np.mean(excess_returns)
    std_dev = np.std(returns)
    sharpe_ratio = mean_excess_return / std_dev
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

def classify_annualized_returns(annualized_returns):
    percentiles = np.percentile(list(annualized_returns.values()), [25, 50, 75])
    classification = []
    
    for return_value in annualized_returns.values():
        if return_value < percentiles[0]:
            classification.append('Low')
        elif return_value < percentiles[1]:
            classification.append('Medium')
        elif return_value < percentiles[2]:
            classification.append('High')
        else:
            classification.append('Very High')
    
    return classification

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

def matching_stocks(classified_stocks, customer_classification):
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
    
    return portfolio

def generate_random_portfolios(matching, not_matching, num_portfolios=1000, min_portfolio_size=1, max_portfolio_size=30):
    random_portfolios = [
        generate_random_portfolio(matching, not_matching, min_portfolio_size, max_portfolio_size)
        for _ in range(num_portfolios)
    ]
    return random_portfolios

def calculate_portfolio_metrics(newdf, random_portfolios, num_portfolios=1000):
    expected_returns_global = {}
    expected_risk_global = {}
    expected_sharpe_ratio_global = {}
    
    for portfolio in random_portfolios:
        portfolio_tuple = tuple(portfolio)
        portfolio_len = len(portfolio)
        
        expected_returns_local = np.zeros(num_portfolios)
        expected_risk_local = np.zeros(num_portfolios)
        expected_sharpe_ratio_local = np.zeros(num_portfolios)
        
        concatenated_data = pd.concat([newdf[stock] for stock in portfolio], axis=1)
        log_returns_local = np.log(concatenated_data / concatenated_data.shift(1))
        
        for k in range(num_portfolios):
            w = np.random.random(portfolio_len)
            w /= np.sum(w)
            
            mean_log_return = log_returns_local.mean()
            sigma = log_returns_local.cov()
            
            expected_returns_local[k] = np.sum(mean_log_return * w)
            expected_risk_local[k] = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
            expected_sharpe_ratio_local[k] = expected_returns_local[k] / expected_risk_local[k]
        
        max_index_local = expected_sharpe_ratio_local.argmax()
        max_sharpe_ratio = expected_sharpe_ratio_local[max_index_local]
        expected_returns_global[portfolio_tuple] = expected_returns_local[max_index_local]
        expected_risk_global[portfolio_tuple] = expected_risk_local[max_index_local]
        expected_sharpe_ratio_global[portfolio_tuple] = max_sharpe_ratio
    
    return expected_returns_global, expected_risk_global, expected_sharpe_ratio_global

def plot_portfolios(expected_returns_global, expected_risk_global, expected_sharpe_ratio_global):
    max_portfolio = max(expected_sharpe_ratio_global, key=expected_sharpe_ratio_global.get)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(
        expected_risk_global.values(),
        expected_returns_global.values(),
        c=list(expected_sharpe_ratio_global.values())
    )
    plt.xlabel('Expected Risk')
    plt.ylabel('Expected Returns')
    plt.colorbar(label='Expected Sharpe Ratio')
    plt.scatter(expected_risk_global[max_portfolio], expected_returns_global[max_portfolio], c='red')
    plt.show()

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
matching, not_matching = matching_stocks(classified_stocks, customer_classification)
random_portfolios = generate_random_portfolios(matching, not_matching)
print("random portfolios:",random_portfolios)
expected_returns_global, expected_risk_global, expected_sharpe_ratio_global = calculate_portfolio_metrics(newdf, random_portfolios)
plot_portfolios(expected_returns_global, expected_risk_global, expected_sharpe_ratio_global)
