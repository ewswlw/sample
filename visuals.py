import pandas as pd
import quantstats as qs

# FUNCTION DEFINITION: To Generate Quant Stat Reports 
def generate_quantstats_report(strategies, title):
    """
    Combine portfolio returns from multiple strategies into a single DataFrame and generate a QuantStats report.

    Parameters:
    - strategies: list of dicts, each containing:
        - 'name': str, the name of the strategy.
        - 'portfolio': Portfolio object with a returns() method.
    - title: str, the title of the QuantStats report and the name of the output HTML file.

    Returns:
    - combined_returns: DataFrame with combined returns of the portfolios.
    """
    # Validate input types
    if not isinstance(strategies, list) or not all(isinstance(s, dict) for s in strategies):
        raise ValueError("strategies must be a list of dictionaries with 'name' and 'portfolio' keys")
    if not isinstance(title, str):
        raise ValueError("title must be a string")

    # Initialize an empty DataFrame for combined returns
    combined_returns = pd.DataFrame()

    # Iterate over each strategy, retrieve returns, and add to combined DataFrame
    for strategy in strategies:
        if 'name' not in strategy or 'portfolio' not in strategy:
            raise ValueError("Each strategy dictionary must contain 'name' and 'portfolio' keys")
        
        strategy_name = strategy['name']
        portfolio = strategy['portfolio']

        # Check if the portfolio has a callable returns() method
        if not hasattr(portfolio, 'returns') or not callable(getattr(portfolio, 'returns')):
            raise ValueError(f"The portfolio for strategy '{strategy_name}' must have a callable returns() method")
        
        # Retrieve returns
        try:
            returns = portfolio.returns()
        except Exception as e:
            raise ValueError(f"Error retrieving returns from portfolio '{strategy_name}': {e}")

        # Ensure returns are a DataFrame or Series
        if not isinstance(returns, (pd.DataFrame, pd.Series)):
            raise ValueError(f"The returns from portfolio '{strategy_name}' must be a DataFrame or Series")

        # Convert Series to DataFrame if necessary
        if isinstance(returns, pd.Series):
            returns = returns.to_frame(strategy_name)
        else:
            returns.columns = [strategy_name]

        # Add returns to the combined DataFrame
        if combined_returns.empty:
            combined_returns = returns
        else:
            combined_returns = pd.concat([combined_returns, returns], axis=1)

    # Check for missing values
    if combined_returns.isnull().values.any():
        print("Warning: Combined returns DataFrame contains missing values. Filling with 0.")
        combined_returns.fillna(0, inplace=True)

    # Generate QuantStats report
    output_filename = f"{title}.html"
    try:
        qs.reports.html(combined_returns, output=output_filename, title=title)
        print(f"QuantStats report generated and saved to {output_filename}")
    except Exception as e:
        raise RuntimeError(f"Error generating QuantStats report: {e}")

    return combined_returns

# Corrected example usage
#strategies = [
#    {'name': 'US IG MA Strategy', 'portfolio': us_ig_portfolio},
#    {'name': 'US HY MA Strategy', 'portfolio': us_hy_portfolio},
#    {'name': 'Buy and Hold Strategy', 'portfolio': buy_and_hold_portfolio}
#]

#generate_quantstats_report(strategies, 'Do US Credit Markets Lead Trends in CAD IG')
