import numpy as np
from fastdtw import fastdtw
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kendalltau
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

class stockAnalyzer:


    def fill_missing_minutes(self, data, Type):
        """
        Introduces NaN values for missing datetime in the 'Close' column of a given DataFrame.
        Args:
            data: The DataFrame containing stock data.
            date: The date in 'YYYY-MM-DD' format.
        Returns:
            A DataFrame with NaN values filled for missing datetime.
        """
        # Create a DatetimeIndex with 1-minute intervals for the pre-market time range
        date_str = data.index[0].strftime('%Y-%m-%d')
        if Type == 'PRE':
            full_index = pd.date_range(start=(f"{date_str} 06:01:00-06:00"), end=(f"{date_str} 09:30:00-06:00"), freq='1min')
        elif Type == 'OPEN':
            full_index = pd.date_range(start=(f"{date_str} 09:30:00-06:00"), end=(f"{date_str} 11:30:00-06:00"), freq='1min')
        # Reindex the DataFrame with the full index
        data = data.reindex(full_index)
        data = data.ffill()

        return data


    def get_pre_market_measures(self, data):
        # Extract the first day's closing prices
        first_day_close = np.asanyarray(self.fill_missing_minutes(data[0]["pre_market_data"]["Close"], 'PRE'))
        first_day_close = pd.Series(first_day_close).rolling(window=5).mean()#
        first_day_close = first_day_close[~np.isnan(first_day_close)]
        normalizer = first_day_close.iloc[-1]
        first_day_close_norm = ((first_day_close-normalizer)/normalizer)*100


        # Initialize an array to store correlations
        pearson = []
        kendall_tau = []
        dynamic_time_warping = []
        mutual_information = []

        # Iterate over the remaining days
        for day_data in data[1:]:
            close_prices = np.asanyarray(self.fill_missing_minutes(day_data["pre_market_data"]["Close"], 'PRE'))
            close_prices = pd.Series(close_prices).rolling(window=5).mean()#
            close_prices = close_prices[~np.isnan(close_prices)]
            normalizer = close_prices.iloc[-1]
            close_prices_norm = ((close_prices-normalizer)/normalizer)*100

            # Calculate and append measures between the first day and the current day
            value = np.corrcoef(first_day_close_norm, close_prices_norm)[0, 1]
            pearson.append(value)
            
            # Apply DTW
            distance, path = fastdtw(first_day_close_norm, close_prices_norm)
            dynamic_time_warping.append(distance)

            # Kendall's Tau correlation
            tau, p_value = kendalltau(first_day_close, close_prices)
            kendall_tau.append(tau)

            # Discretize the continuous signals
            # Use KBinsDiscretizer to bin the signals into discrete categories
            discretizer = KBinsDiscretizer(n_bins=30, encode='ordinal', strategy='uniform', subsample=200000)
            first_day_close_discrete = discretizer.fit_transform(np.asanyarray(first_day_close_norm).reshape(-1, 1)).flatten()
            close_prices_discrete = discretizer.fit_transform(np.asanyarray(close_prices_norm).reshape(-1, 1)).flatten()
            
            # Calculate Mutual Information
            mutual_information.append( mutual_info_score(first_day_close_discrete, close_prices_discrete) )

        return [pearson, kendall_tau, dynamic_time_warping, mutual_information]
    
    def max_min_of_abs(self, data_list, Type):
        """
        Finds the indices and values of the 3 largest and 3 smallest absolute values in a list.

        Args:
            data_list: A list of numerical values.

        Returns:
            A tuple containing two lists:
            - max_values: A list of tuples containing the indices and values of the 3 largest absolute values.
            - min_values: A list of tuples containing the indices and values of the 3 smallest absolute values.
        """
        abs_values = [abs(x) for x in data_list]
        
        # Get indices of the 3 largest and 3 smallest absolute values
        max_indices = sorted(range(len(abs_values)), key=lambda i: abs_values[i], reverse=True)[:3]
        min_indices = sorted(range(len(abs_values)), key=lambda i: abs_values[i])[:3]

        # Get the corresponding values
        max_values = [(index, data_list[index]) for index in max_indices]
        min_values = [(index, data_list[index]) for index in min_indices]

        # Conditional return
        if Type == 'Max':
            values = max_values
        elif Type == 'Min':
            values = min_values
        return values