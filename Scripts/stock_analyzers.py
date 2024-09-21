import numpy as np
from fastdtw import fastdtw
import pandas as pd
import datetime
import pytz
from scipy.stats import kendalltau
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import ConstantKernel as C
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

class stockAnalyzer:
    def fill_missing_minutes(self, data, Type, adjustment='11:30'):
        """
        Introduces NaN values for missing datetime in the 'Close' column of a given DataFrame.
        Args:
            data: The DataFrame containing stock data.
            date: The date in 'YYYY-MM-DD' format.
        Returns:
            A DataFrame with missing values filled with the nearest data.
        """
        # Create a DatetimeIndex with 1-minute intervals for the pre-market time range
        date_str = data.index[0].strftime('%Y-%m-%d')
        if Type == 'PRE':
            full_index = pd.date_range(start=(f"{date_str} 06:01:00-04:00"), end=(f"{date_str} 09:29:00-04:00"), freq='1min')
        elif Type == 'OPEN':
            full_index = pd.date_range(start=(f"{date_str} 09:30:00-04:00"), end=(f"{date_str} 16:00:00-04:00"), freq='1min')
        elif Type == 'ACTIVE':
            full_index = pd.date_range(start=(f"{date_str} 09:30:00-04:00"), end=(f"{date_str} {adjustment}:00-04:00"), freq='1min')
            #full_index = pd.date_range(start=(f"{date_str} 09:30:00-04:00"), end=(f"{date_str} 16:00:00-04:00"), freq='1min')# For testing purposes
        # Reindex the DataFrame with the full index
        data = data.reindex(full_index)
        # Special case: fill the first missing values by bfill()
        if data.isna().iloc[0]:
            data = data.bfill()
        
        # Fill remaining missing values by ffill()
        data = data.ffill()

        return data

    def get_pre_market_measures(self, data, smoothing_window, matching_window=30):
        # Extract the first day's closing prices
        first_day_close = np.asanyarray(self.fill_missing_minutes(data[0]["pre_market_data"]["Close"], 'PRE'))

        time_indexes = data[0]["pre_market_data"].index
        first_time = time_indexes[0].time()
        # Calculate the difference in minutes between the first time and '09:29:00'
        target_time = datetime.datetime.strptime('09:29:00', '%H:%M:%S').time()
        first_day_time_diff = int((datetime.datetime.combine(datetime.datetime.today(), target_time) - datetime.datetime.combine(datetime.datetime.today(), first_time)).total_seconds() / 60)

        first_day_close = pd.Series(first_day_close).rolling(window=smoothing_window).mean()#
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

            time_indexes = day_data["pre_market_data"].index
            first_time = time_indexes[0].time()
            # Calculate the difference in minutes between the first time and '09:29:00'
            close_prices_time_diff = int((datetime.datetime.combine(datetime.datetime.today(), target_time) - datetime.datetime.combine(datetime.datetime.today(), first_time)).total_seconds() / 60)


            close_prices = pd.Series(close_prices).rolling(window=smoothing_window).mean()#
            close_prices = close_prices[~np.isnan(close_prices)]
            normalizer = close_prices.iloc[-1]
            close_prices_norm = ((close_prices-normalizer)/normalizer)*100

            # Ensure both arrays have the same length
            min_length = min(first_day_time_diff, close_prices_time_diff, abs(matching_window))
            first_day_close_norm = first_day_close_norm[-min_length+1:]
            close_prices_norm = close_prices_norm[-min_length+1:]

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
        # Get the absolute values of the data
        abs_values = [abs(x) for x in data_list]

         # Conditional return
        if Type == 'Max':
            abs_values = [0 if np.isnan(x) else x for x in abs_values]
        elif Type == 'Min':
            abs_values = [10000 if np.isnan(x) else x for x in abs_values]

        
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

class stockPredictor:
    #Create object with the best 3 predictors for each type of metric
    def __init__(self, predictors, to_be_predicted):
        self.a = np.asanyarray(predictors[0])
        self.b = np.asanyarray(predictors[1])
        self.c = np.asanyarray(predictors[2])
        self.x = np.asanyarray(to_be_predicted)

    def data_divider(self, where_to_cut):
        # Store the cut-off point
        self.where_to_cut = where_to_cut
        
        # Determine the minimum value between the cut-off point and the length of x
        N = min(where_to_cut, len(self.x))
        
        # Create a feature matrix X by combining the first N elements of a, b, and c
        self.X = np.column_stack((self.a[:N], self.b[:N], self.c[:N]))

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

    def DTW_regresion(self):
        N = self.where_to_cut
        # Compute DTW distances between x and a, b, c
        aligned_a = self.align_series_with_dtw(self.X[:, 0], self.x[:N])
        aligned_b = self.align_series_with_dtw(self.X[:, 1], self.x[:N])
        aligned_c = self.align_series_with_dtw(self.X[:, 2], self.x[:N])
        # Combine the aligned series into a feature matrix
        X_aligned = np.column_stack((aligned_a, aligned_b, aligned_c))
        
        # Apply StandardScaler to X_aligned and x
        scaler_X_aligned = StandardScaler().fit(X_aligned)
        X_aligned_scaled = scaler_X_aligned.transform(X_aligned)
        scaler_x = StandardScaler().fit(self.x[:N].reshape(-1, 1))
        x_scaled = scaler_x.transform(self.x[:N].reshape(-1, 1)).flatten()
        
        # Train a linear regression model to predict x from aligned series
        model = LinearRegression().fit(X_aligned_scaled, x_scaled)
        
        # Use the trained model to predict the future values of x
        scaler_X = StandardScaler().fit(np.column_stack((self.a, self.b, self.c)))
        X_all_scaled = scaler_X.transform(np.column_stack((self.a, self.b, self.c)))
        
        # Predict the future values of x
        x_future_pred_scaled = model.predict(X_all_scaled)
        
        # Inverse transform the prediction to get it back to the original scale
        prediction = scaler_x.inverse_transform(x_future_pred_scaled.reshape(-1, 1)).flatten()
        
        return prediction

    def ridge_model(self):
        '''Fit a Ridge regression model to handle multicollinearity'''
        # Apply StandardScaler to X and x
        scaler_X = StandardScaler().fit(self.X)
        X_scaled = scaler_X.transform(self.X)
        scaler_x = StandardScaler().fit(self.x[:self.where_to_cut].reshape(-1, 1))
        x_scaled = scaler_x.transform(self.x[:self.where_to_cut].reshape(-1, 1)).flatten()
        
        # Fit the Ridge regression model
        ridge_model = Ridge(alpha=0.1).fit(X_scaled, x_scaled)  # Alpha is the regularization strength
        
        # Use the trained model to predict the future values of x
        X_all_scaled = scaler_X.transform(np.column_stack((self.a, self.b, self.c)))
        prediction_scaled = ridge_model.predict(X_all_scaled)
        
        # Inverse transform the prediction to get it back to the original scale
        prediction = scaler_x.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()
        
        return prediction

    def setar_model(self):
        '''Fit a Self-Exciting Threshold Autoregressive (SETAR) model'''

        # Prepare data
        x = self.x
        X = self.X

        N = min(len(self.x), len(self.a), len(self.b), len(self.c))

        where_to_cut = self.where_to_cut

        # Apply StandardScaler to X and x (up to where_to_cut)
        scaler_X = StandardScaler().fit(X[:where_to_cut])
        X_scaled = scaler_X.transform(X[:where_to_cut])
        scaler_x = StandardScaler().fit(x[:where_to_cut].reshape(-1, 1))
        x_scaled = scaler_x.transform(x[:where_to_cut].reshape(-1, 1)).flatten()

        # Compute lagged values of x_scaled
        x_lagged = x_scaled[:-1]  # x at time t-1
        x_current = x_scaled[1:]  # x at time t
        X_current = X_scaled[1:, :]  # Corresponding X at time t

        # Adjust where_to_cut since we lost one data point due to lagging
        where_to_cut_adjusted = where_to_cut - 1

        # Determine the threshold using the median of x_lagged up to where_to_cut_adjusted
        threshold = np.median(x_lagged[:where_to_cut_adjusted])

        # Split data into regimes based on the threshold
        regime1_indices = np.where(x_lagged[:where_to_cut_adjusted] <= threshold)[0]
        regime2_indices = np.where(x_lagged[:where_to_cut_adjusted] > threshold)[0]

        # Prepare data for each regime
        X_regime1 = X_current[regime1_indices]
        y_regime1 = x_current[regime1_indices]

        X_regime2 = X_current[regime2_indices]
        y_regime2 = x_current[regime2_indices]

        # Fit Linear Regression models for each regime
        model_regime1 = LinearRegression().fit(X_regime1, y_regime1)
        model_regime2 = LinearRegression().fit(X_regime2, y_regime2)

        # Initialize an array to store scaled predictions
        predictions_scaled = np.zeros(N)
        predictions_scaled[0] = np.nan  # First value has no prediction due to lag

        # Predict values iteratively
        for t in range(1, N):
            if t < where_to_cut:
                # Use actual lagged x value
                x_lagged_t = x_scaled[t - 1]
                X_t = X_scaled[t, :].reshape(1, -1)
            else:
                # Use predicted lagged x value
                x_lagged_t = predictions_scaled[t - 1]
                X_t_original = np.array([self.a[t], self.b[t], self.c[t]]).reshape(1, -1)
                X_t = scaler_X.transform(X_t_original)

            if np.isnan(x_lagged_t):
                # Cannot make prediction without lagged x
                predictions_scaled[t] = np.nan
                continue

            # Select the appropriate model based on the threshold
            if x_lagged_t <= threshold:
                x_t_pred = model_regime1.predict(X_t)[0]
            else:
                x_t_pred = model_regime2.predict(X_t)[0]

            predictions_scaled[t] = x_t_pred

        # Inverse transform the scaled predictions to get them back to the original scale
        predictions = scaler_x.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

        return predictions
        
    
    def elastic_net(self):
        '''Fit an Elastic Net model with standardized data'''
        # Apply StandardScaler to X and x
        scaler_X = StandardScaler().fit(self.X)
        X_scaled = scaler_X.transform(self.X)
        scaler_x = StandardScaler().fit(self.x[:self.where_to_cut].reshape(-1, 1))
        x_scaled = scaler_x.transform(self.x[:self.where_to_cut].reshape(-1, 1)).flatten()
        
        # Fit the Elastic Net model
        model = ElasticNet(alpha=0.1, l1_ratio=0.2)  # Customize alpha and l1_ratio as needed
        model.fit(X_scaled, x_scaled)  # Fit the model
        
        # Use the trained model to predict the future values of x
        X_all_scaled = scaler_X.transform(np.column_stack((self.a, self.b, self.c)))
        prediction_scaled = model.predict(X_all_scaled)
        
        # Inverse transform the prediction to get it back to the original scale
        prediction = scaler_x.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()
        
        return prediction
    
class stockNormalizer:
    def normalize_market_data(self, day_data, ventana, special_case=False):
        # Normalize pre-market data
        SA = stockAnalyzer()
        pre_data_raw = SA.fill_missing_minutes(day_data["pre_market_data"]["Close"], 'PRE')
        pre_data = day_data["pre_market_data"]["Close"].rolling(window=ventana).mean()
        pre_normalizer = pre_data_raw.iloc[-1]
        pre_data = pre_data.dropna()
        normalized_pre_data = ((pre_data - pre_normalizer) / pre_normalizer) * 100

        # Normalize open market data
        if special_case:
            nyc_timezone = pytz.timezone('America/New_York')
            now = datetime.datetime.now(nyc_timezone)
            now = f"{now.strftime('%H:%M')}"
            open_data_raw = SA.fill_missing_minutes(day_data["open_market_data"]["Open"], 'ACTIVE', now)
        else:
            open_data_raw = SA.fill_missing_minutes(day_data["open_market_data"]["Open"], 'OPEN')
        open_data = open_data_raw.rolling(window=ventana).mean().shift(-(ventana))
        open_normalizer = open_data_raw.iloc[0]
        open_data = open_data.dropna()
        normalized_open_data = ((open_data - open_normalizer) / open_normalizer) * 100
        return normalized_pre_data, normalized_open_data
    
    