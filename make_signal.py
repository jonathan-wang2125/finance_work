import pandas as pd
import statsmodels.api as sm
import time

# general idea: groupby factor, groupby rolling window
# then grab the right returns (i.e tuesdays) after forming them
# the cumulate


def get_all_factors(df):
    
    unique_names_set = set(df['name'].unique())
    return unique_names_set


def get_next_instance_date(df, s, seasonality_type, t, nth=1):
    valid_seasonality_types = {
        'weekday': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
        'day of month': list(range(1, 32)), 
        'month of year': ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'],
        'nth weekday of month': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    }
    
    t = pd.to_datetime(t)
    if seasonality_type == 'weekday':
        next_date = t + pd.DateOffset(days=(7 - t.weekday() + valid_seasonality_types[seasonality_type].index(s.lower())) % 7)
    elif seasonality_type == 'day of month':
        next_month = t.replace(day=int(s)) + pd.DateOffset(months=1)
        next_date = next_month.replace(day=int(s))
        if next_date <= t:
            next_date = next_month.replace(day=int(s))
    elif seasonality_type == 'month of year':
        months = valid_seasonality_types[seasonality_type]
        next_month_index = (months.index(s.lower()) + 1) % len(months)
        next_date = t.replace(month=next_month_index + 1, day=1)
        if next_date <= t:
            next_date = t.replace(year=t.year + 1, month=next_month_index + 1, day=1)
    elif seasonality_type == 'nth weekday of month':
        next_date = get_nth_weekday_of_month(df, s, nth).date.max() + pd.DateOffset(months=1)
    return next_date

def validate_inputs(s, seasonality_type):
    seasonality_type = seasonality_type.lower()
    valid_seasonality_types = {
        'weekday': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
        'day of month': list(range(1, 32)), 
        'month of year': ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'],
        'nth weekday of month': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    }
    if seasonality_type not in valid_seasonality_types:
            return "Invalid seasonality type"
    if seasonality_type == 'nth weekday of month' and s.lower() not in valid_seasonality_types[seasonality_type]:
        return f"Invalid {seasonality_type}"
        
    if isinstance(s, str) and s.lower() not in valid_seasonality_types[seasonality_type]:
        return f"Invalid {seasonality_type}"
        
    if isinstance(s, int) and s not in valid_seasonality_types[seasonality_type]:
        return f"Invalid {seasonality_type}"
        
    return None




def get_nth_weekday_of_month(df, weekday, nth):
    """
    Helper function to get the nth occurrence of a specific weekday within each month.
    """
    weekday = weekday.lower()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.strftime('%A').str.lower()
    df_filtered = df[df['weekday'] == weekday]
    df_filtered = df_filtered.sort_values(by='date')
    df_filtered['rank'] = df_filtered.groupby(['year', 'month']).cumcount() + 1
    df_nth_weekday = df_filtered[df_filtered['rank'] == nth]
    return df_nth_weekday[['date', 'ret']]

def form_signal(df, factor, s, seasonality_type, t="2023-12-21", k='1Y', nth=1):
    """
    factor: string name of factor which we seek to find returns
    df: DataFrame containing the data 
    s: specific day of week (e.g., 'monday', 'tuesday', etc.), day of month (0-30), or month of year ('january', 'february', etc.)
    t: specific end date for the window (YYYY-MM-DD)
    k: window size for calculating returns (e.g., '1Y' for one year)
    seasonality_type: 'weekday', 'day of month', 'month of year', or 'nth weekday of month'
    nth: the nth occurrence of the weekday within the month (only applicable if seasonality_type is 'nth weekday of month')
    """
    t = pd.to_datetime(t)
    # Define the rolling window size
    if k.endswith('Y'):
        window_size = pd.DateOffset(years=int(k[:-1]))
    elif k.endswith('M'):
        window_size = pd.DateOffset(months=int(k[:-1]))
    elif k.endswith('D'):
        window_size = pd.DateOffset(days=int(k[:-1]))

    # Convert 'date' column in DataFrame to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')


    # Filter the DataFrame for the given factor and date range
    df_factor = df[(df['name'] == factor) & (df['date'] <= t)]
    
    # Define filtering logic based on seasonality_type
    filter_logic = {
        'weekday': lambda df: df[df['date'].dt.strftime('%A').str.lower() == s.lower()],
        'day of month': lambda df: df[df['date'].dt.day == int(s)],
        'month of year': lambda df: df[df['date'].dt.month_name().str.lower() == s.lower()],
        'nth weekday of month': lambda df: get_nth_weekday_of_month(df, s, nth)
    }
    
    if seasonality_type in filter_logic:
        df_seasonal = filter_logic[seasonality_type](df_factor)
    else:
        raise ValueError("Invalid seasonality_type. Must be 'weekday', 'day of month', 'month of year', or 'nth weekday of month'.")

    # Ensure the data is sorted by date
    df_seasonal = df_seasonal.sort_values(by='date')

    # Create a rolling window of cumulative returns
    cumulative_returns = []
    cumulative_returns = []
    for date in df_seasonal['date']:
        if date <= t:
            start_date = date - window_size
            window_data = df_seasonal[(df_seasonal['date'] > start_date) & (df_seasonal['date'] <= date)]
            if not window_data.empty:
                cumulative_return = (1 + window_data['ret']).prod() - 1
                cumulative_returns.append(cumulative_return)
            else:
                cumulative_returns.append(None)
    
    df_seasonal['cumulative_return'] = cumulative_returns

    df_seasonal = df_seasonal.dropna(subset=['cumulative_return'])

    drop_date = df_seasonal['date'].min() + window_size
    df_seasonal = df_seasonal[df_seasonal['date'] > drop_date]

    return df_seasonal[['date', 'cumulative_return']]

def calculate_naive_return(df, factor, s, seasonality_type, t="2023-12-21", k='1Y', nth=1):
    """
    Calculate rolling cumulative returns for a given factor over specified seasonality type.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data with columns 'date', 'name', and 'ret'.
    factor (str): Name of the factor for which to calculate returns.
    s (str/int): Specific day of week (e.g., 'monday', 'tuesday', etc.), day of month (1-31), or month of year ('january', 'february', etc.).
    seasonality_type (str): 'weekday', 'day of month', 'month of year', or 'nth weekday of month'.
    t (str): Specific end date for the window (YYYY-MM-DD).
    k (str): Window size for calculating cumulative returns (e.g., '1Y' for one year).
    g (int): Rolling window size for calculating cumulative returns (number of periods in each rolling window).
    nth (int): The nth occurrence of the weekday within the month (only applicable if seasonality_type is 'nth weekday of month').
    
    Returns:
    pd.DataFrame: DataFrame with columns 'date' and 'cumulative_return' representing the rolling cumulative returns.
    """
    
    t = pd.to_datetime(t)
    if k.endswith('Y'):
        window_size = pd.DateOffset(years=int(k[:-1]))
    elif k.endswith('M'):
        window_size = pd.DateOffset(months=int(k[:-1]))
    elif k.endswith('D'):
        window_size = pd.DateOffset(days=int(k[:-1]))

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    
    
    df_factor = df[df['name'] == factor].copy()
    
    def get_nth_weekday_of_month(df, weekday, nth):
        def is_nth_weekday_of_month(date):
            first_day_of_month = date.replace(day=1)
            nth_weekday_date = first_day_of_month + pd.DateOffset(weeks=nth-1)
            while nth_weekday_date.strftime('%A').lower() != weekday.lower():
                nth_weekday_date += pd.DateOffset(days=1)
            return date == nth_weekday_date

        return df[df['date'].apply(is_nth_weekday_of_month)]

    df_factor = df_factor.sort_values(by='date')
    
    if seasonality_type == 'weekday':
        dates_of_interest = (df_factor[df_factor['date'].dt.strftime('%A').str.lower() == s.lower()]['date']).copy()
    elif seasonality_type == 'day of month':
        dates_of_interest = (df_factor[df_factor['date'].dt.day == int(s)]['date']).copy()
    elif seasonality_type == 'month of year':
        dates_of_interest = (df_factor[df_factor['date'].dt.month_name().str.lower() == s.lower()]['date']).copy()
    elif seasonality_type == 'nth weekday of month':
        dates_of_interest = (get_nth_weekday_of_month(df_factor, s, nth)['date']).copy()
    else:
        raise ValueError("Invalid seasonality_type. Must be 'weekday', 'day of month', 'month of year', or 'nth weekday of month'.")
    
    rolling_cumulative_returns = []


    for date in dates_of_interest:
        if date <= t:
            start_date = date - window_size
            window = df_factor[(df_factor['date'] > start_date) & (df_factor['date'] <= date)]
            if len(window) > 0:
                cumulative_return = (1 + window['ret']).prod() - 1
                rolling_cumulative_returns.append((date, cumulative_return))
    
    result_df = pd.DataFrame(rolling_cumulative_returns, columns=['date', 'cumulative_return'])
    
    earliest_date_with_return = result_df['date'].min() + window_size
    result_df = result_df[result_df['date'] >= earliest_date_with_return]
    
    return result_df


def test_signal(df, factor, s, seasonality_type,  t="2023-12-21", k='1Y', nth=1):
    """
    Evaluate regression of future returns compared to past returns 
    on signal returns and naive returns

    Parameters:
    df (pd.DataFrame): DataFrame containing the data with columns 'date', 'name', and 'ret'.
    factor (str): Name of the factor for which to calculate returns.
    s (str/int): Specific day of week (e.g., 'monday', 'tuesday', etc.), day of month (1-31), or month of year ('january', 'february', etc.).
    seasonality_type (str): 'weekday', 'day of month', 'month of year', or 'nth weekday of month'.
    t (str): date to evaluate from. Enter YYYY-MM-DD
    k (str): Length of backwards window either years in ter 1Y, 2Y, etc or number of days STILL A STRING
    l (str): Length of Forward window either years in ter 1Y, 2Y, etc or number of days STILL A STRING
    g (int): Rolling window size for calculating cumulative returns (number of periods in each rolling window).
    nth (int): The nth occurrence of the weekday within the month (only applicable if seasonality_type is 'nth weekday of month').
    """
    
    validation_results = validate_inputs(s, seasonality_type)
    if validation_results:
        return validation_results

    # Generate the signal DataFrame
    signal = form_signal(df, factor, s, seasonality_type, t, k, nth)
    signal.rename(columns={'cumulative_return': 'signal'}, inplace=True)

    df_factor = df[(df['name'] == factor) & (df['date'] <= t)]
    
    filter_logic = {
        'weekday': lambda df: df[df['date'].dt.strftime('%A').str.lower() == s.lower()],
        'day of month': lambda df: df[df['date'].dt.day == int(s)],
        'month of year': lambda df: df[df['date'].dt.month_name().str.lower() == s.lower()],
        'nth weekday of month': lambda df: get_nth_weekday_of_month(df, s, nth)
    }
    
    if seasonality_type in filter_logic:
        df_seasonal = filter_logic[seasonality_type](df_factor)
    else:
        raise ValueError("Invalid seasonality_type. Must be 'weekday', 'day of month', 'month of year', or 'nth weekday of month'.")

    df_seasonal = df_seasonal.sort_values(by='date')

    # Ensure the signal DataFrame is sorted by date to align with df_seasonal
    signal = signal.sort_values(by='date')

    # Merge signal with df_seasonal
    merged_df = pd.merge(signal, df_seasonal, on='date', how='inner')
    
    if merged_df.empty:
        return "No matching dates for regression"
    
    # Regression of df_seasonal against signal
    X_seasonal = merged_df[['ret']]  # Independent variable (Seasonal returns)
    y_signal = merged_df['signal']  # Dependent variable (Signal returns)
    
    X_seasonal = sm.add_constant(X_seasonal)  # Add constant term to the predictor
    model_signal = sm.OLS(y_signal, X_seasonal).fit()
    
    alpha_signal = model_signal.params['const']
    beta_signal = model_signal.params['ret']
    
    # Calculate naive returns for regression
    naive_returns = calculate_naive_return(df, factor, s, seasonality_type, t, k, nth)
    
    # Merge naive returns with df_seasonal for regression
    naive_returns.rename(columns={'cumulative_return': 'naive_return'}, inplace=True)
    merged_df_naive = pd.merge(df_seasonal[['date', 'ret']], naive_returns, on='date', how='inner')
    
    if merged_df_naive.empty:
        return "No matching dates for naive return regression"
    
    X_naive = merged_df_naive[['ret']]  # Independent variable (Seasonal returns)
    y_naive = merged_df_naive['naive_return']  # Dependent variable (Naive returns)
    
    X_naive = sm.add_constant(X_naive)  # Add constant term to the predictor
    model_naive = sm.OLS(y_naive, X_naive).fit()
    
    alpha_naive = model_naive.params['const']
    beta_naive = model_naive.params['ret']
    
    # Create DataFrame with the required columns
    result_df = merged_df[['date', 'signal']].copy()
    result_df['signal_alpha'] = alpha_signal
    result_df['signal_beta'] = beta_signal
    result_df = result_df.merge(naive_returns[['date', 'naive_return']], on='date', how='left')
    result_df['naive_alpha'] = alpha_naive
    result_df['naive_beta'] = beta_naive
    
    result_df.rename(columns={'naive_return': 'naive'}, inplace=True)
    
    return result_df
    


    
    
  




df = pd.read_csv('/Users/jonathan/Desktop/seasonality_data/cleaned_data.csv')

if __name__ == '__main__':




    # test_signal_result = test_signal(df, 'age', 'tuesday', 'weekday', '2023-01-01', '1Y', nth = 1)
    # print(test_signal_result)

    print(((get_all_factors(df))))
    





