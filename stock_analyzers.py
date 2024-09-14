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

from fastdtw import fastdtw
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge

class stockPredictor:
    #Create object with the best 3 predictors for each type of metric
    def __init__(self, predictor_1, predictor_2, predictor_3, to_be_predicted):
        self.a = predictor_1
        self.b = predictor_2
        self.c = predictor_3
        self.x = to_be_predicted

    def data_divider(self,where_to_cut):
        N = where_to_cut
        self.X = np.column_stack((self.a[:N], self.b[:N], self.c[:N]))
        self.Y = np.column_stack((self.a[N:], self.b[N:], self.c[N:]))

    def align_series_with_dtw(self, source, target):
        '''Perform DTW to get the path of alignment'''
        _, path = fastdtw(source, target)
        # Create an aligned version of the source series with the same length as the target
        aligned_source = np.zeros(len(target))  # Initialize with the same length as target
        # Use the DTW path to align source to target
        for i, (source_idx, target_idx) in enumerate(path):
            # Place the source value at the position corresponding to the target index
            aligned_source[target_idx] = source[source_idx]
        # Some target indices may not be assigned, so we interpolate them
        for i in range(1, len(aligned_source)):
            if aligned_source[i] == 0:  # Check if any values are missing
                aligned_source[i] = aligned_source[i - 1]  # Fill missing values by repeating the previous
        return aligned_source

    def DTW_regresion(self, where_to_cut):
        N = where_to_cut
        # Compute DTW distances between x and a, b, c
        aligned_a = self.align_series_with_dtw(self.a[:N], self.x[:N])
        aligned_b = self.align_series_with_dtw(self.b[:N], self.x[:N])
        aligned_c = self.align_series_with_dtw(self.c[:N], self.x[:N])
        # Combine the aligned series into a feature matrix
        X_aligned = np.column_stack((aligned_a, aligned_b, aligned_c))
        # Train a linear regression model to predict x from aligned series
        model = LinearRegression().fit(X_aligned, self.x[:N])

        # Stack the future values of a, b, and c (no alignment with future x is needed)
        X_future = self.Y
        X_past = self.X

        # Predict the future values of x
        x_past_pred = model.predict(X_past)
        x_future_pred = model.predict(X_future)
        prediction = np.append(x_past_pred, x_future_pred)
        return prediction

    def ridge_model(self, where_to_cut):
        '''Fit a Ridge regression model to handle multicollinearity'''
        self.data_divider(where_to_cut)
        ridge_model = Ridge(alpha=1.0).fit(self.X, self.x[:where_to_cut])  # Alpha is the regularization strength
        coef_LC = ridge_model.coef_
        prediction = (coef_LC[0]*self.a + coef_LC[1]*self.b + coef_LC[2]*self.c)
        return prediction
    
    def elastic_net(self, where_to_cut):
        '''Fit a Elastic Net model'''
        self.data_divider(where_to_cut)
        model = ElasticNet(alpha=0.25, l1_ratio=0.5)  # Customize alpha and l1_ratio as needed
        model.fit(self.X, self.x[0:where_to_cut])  # Fit the model
        coef_EN = model.coef_  # Retrieve the coefficients (alpha, beta, gamma)
        prediction = (coef_EN[0]*self.a + coef_EN[1]*self.b + coef_EN[2]*self.c)
        return prediction
    