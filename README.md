# Portfolio_Recommendation
Project Overview: Deep Learning-Driven Investment Recommendation with Modern Portfolio Theory
This project ventures beyond traditional investment analysis by combining the power of Deep Learning with Modern Portfolio Theory (MPT). It leverages Deep Learning models to extract insights from historical data and user behavior, while MPT provides a framework for constructing optimal portfolios based on risk-return considerations.

Systems architecture Diagram:
![image](https://github.com/user-attachments/assets/917e2c53-e8e1-42b7-8cb7-ff10880e2560)




Functionalities:

Data Acquisition and Preprocessing:
Utilize appropriate data sources (financial databases, user interactions) and cleaning techniques to prepare data suitable for Deep Learning models.

Deep Learning Model Development:
Train Deep Learning models (Recurrent Neural Networks) to forecast future investment returns or identify potentially profitable investment opportunities.
Functions for model training and evaluation would be included here (not explicitly mentioned earlier).

Modern Portfolio Theory Integration:
calculate_annualized_returns(newdf, years=5): Calculates annualized returns for investments, incorporating Deep Learning model predictions.
calculate_variance(returns): Estimates the variance of daily returns for each investment, still relevant for risk assessment.
calculate_sharpe_ratio(returns, rf=0.02): Maintains the Sharpe Ratio calculation for risk-adjusted return analysis.
Customer Classification (potentially enhanced):
classify_customer(expected_returns, initial_investment, time_period, annualized_returns): Classifies the user based on their desired return, initial investment, investment horizon (risk tolerance), and potentially incorporate Deep Learning model predictions.

Portfolio Generation and Analysis:
matching_stocks(classified_stocks, customer_classification): Identifies investments matching the user's classification (return and risk).
generate_random_portfolio(matching_stocks, other_stocks, min_portfolio_size, max_portfolio_size): Generates a random portfolio with a specified size, considering both matching and non-matching investments.
generate_random_portfolios(matching, not_matching, num_portfolios=1000, min_portfolio_size=1, max_portfolio_size=30): Creates a set of random portfolios for further analysis.
calculate_portfolio_metrics(newdf, random_portfolios, num_portfolios=1000): Estimates expected returns, risk, and Sharpe Ratio for each random portfolio.
plot_portfolios(expected_returns_global, expected_risk_global, expected_sharpe_ratio_global): Visualizes the efficient frontier to aid in selecting portfolios with optimal risk-return trade-offs.
Benefits:

Deep Learning models can potentially capture complex patterns and relationships in historical data, potentially leading to more accurate return predictions.
MPT provides a structured approach to portfolio construction, ensuring diversification and risk management.
The combined approach can offer a more robust and data-driven investment recommendation system.

//what i could have done differently:
1.using LSTM for price prediction was really dumb
2.could have replaced MPT with an genetic algorithm in a way combining both genetic algorithm and neural network for better portfolio reutrns



