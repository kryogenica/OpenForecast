import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pandas_market_calendars as mcal

class stockChecker:
    def is_valid_stock(self, stock_symbol):
        """
        Checks if the given string is a valid stock symbol on Yahoo Finance.

        Args:
          symbol: The stock symbol to check.

        Returns:
          True if the symbol is valid, False otherwise.
        """
        try:
            stock_info = yf.Ticker(stock_symbol).info
            if len(stock_info) > 20:
                return True
            else:
                return False
        except Exception as e:
            print(f"Error checking stock symbol {stock_symbol}: {e}")
            return False

    def stock_has_data(self, stock_symbol):
        '''Verifies if the stock is in an exchage with very little data'''
        excluded_exchanges = {
            'PNK': 'Pink Sheets',
            'OTC': 'Over-The-Counter',
            'OTCQB': 'OTC Markets Group - Venture Market',
            'OTCQX': 'OTC Markets Group - Best Market',
            'GREY': 'Grey Market',
            'NCM': 'Non-Clearing Member',  # Optional: Exclude if desired
        }
        try:
            exchange = yf.Ticker(stock_symbol).info['exchange']
            if exchange in excluded_exchanges:
                exchange_full_name = excluded_exchanges[exchange]
                return False, exchange_full_name
            else:
                return True, []
        except Exception as e:
            print(f"Error checking stock symbol {stock_symbol}: {e}")
            return False

    def is_it_a_trading_day(self, date_to_check):
        '''
        This function checks if a given date is a trading day in the NYSE.
        Provided date should be in the format 'YYYY-MM-DD'
        '''
        # Get the NYSE calendar
        nyse = mcal.get_calendar('NYSE')

        # Get the trading schedule for the day
        schedule = nyse.schedule(start_date=date_to_check, end_date=date_to_check)
        return not schedule.empty       

    def get_filtered_stock_data(self, ticker: str, day: str):
        """
        Fetch and filter stock data for a given ticker on a specific day.
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL').
            day: The day of interest in 'YYYY-MM-DD' format.
        Returns:
            A dictionary with filtered pre-market and open market data.
        """
        # Create the stock object
        stock = yf.Ticker(ticker)

        # Fetch full intraday data for the day with 1-minute resolution
        full_day_data_1m = stock.history(start=day, end=(datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"), interval="1m")
        full_day_data_1m_pre = stock.history(start=day, end=(datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"), interval="1m", prepost=True)

        # Convert the index to a datetime index for filtering
        full_day_data_1m.index = pd.to_datetime(full_day_data_1m.index)
        full_day_data_1m_pre.index = pd.to_datetime(full_day_data_1m_pre.index)

        # Filter pre-market data (6:00 AM to 9:30 AM) at 1-minute intervals
        pre_market_data = full_day_data_1m_pre.between_time("06:01", "09:29")

        # Filter open market data (9:30 AM to 10:30 AM) at 1-minute intervals
        open_market_data = full_day_data_1m.between_time("09:30", "12:41")

        return {
            "pre_market_data": pre_market_data,
            "open_market_data": open_market_data
        }
    
    def get_last_trading_days(self, ticker: str, date, back_window=30):
        """
        Fetch and filter stock data for the last 30 trading days (Monday to Friday).
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL').
        Returns:
            A list of dictionaries containing filtered stock data for each day.
        """
        today = datetime.strptime(date, '%Y-%m-%d').date()

        # To store the filtered data for the last 10 trading days
        historical_data = []

        # Go back and find the last 10 trading days (excluding weekends)
        current_date = today
        counter = 0
        while counter < back_window:
            # Check if it's a weekday (Monday-Friday)
            if current_date.weekday() < 5:  # Monday=0, Friday=4
                if self.is_it_a_trading_day(current_date):
                    day_str = current_date.strftime('%Y-%m-%d')
                    try:
                        # Get filtered data for this day
                        day_data = self.get_filtered_stock_data(ticker, day_str)

                        # If data is returned, append it
                        if not day_data["pre_market_data"].empty and not day_data["open_market_data"].empty:
                            historical_data.append({
                                "date": day_str,
                                "pre_market_data": day_data["pre_market_data"],
                                "open_market_data": day_data["open_market_data"]
                            })
                    except Exception as e:
                        # In case there's an issue (e.g., a holiday with no data), skip this day
                        print(f"Skipping {day_str}: {str(e)}")

            # Move to the previous day
            current_date -= timedelta(days=1)
            counter += 1

        return historical_data
    
    def highlight_col(self, value, dates):
        colors = ['red', 'blue', 'violet']
        for j, i in enumerate(dates):
            if value == dates[j]:
                return f'background-color: {colors[j]}'
            
    def get_day_of_week(self, date_str):
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return days[date_obj.weekday()]

    def get_details_on_stock_per_date(self, data, dates, current):
        '''This function recieves the indexese of the stock and returns a dataframe with opening and closing prices along with the volume'''
        # Initialize an empty list to store the rows of the dataframe
        rows = []
        symbol = ['▲', '⬟', '■', 'Latest']
        if current==True:
            dia = dates[-1]
            try:
                pre_close_price = day_data["pre_market_data"].loc[str(dia)+' 09:29:00-04:00']['Close']
            except KeyError:
                pre_close_price = 'NaN'
            
            try:
                pre_high_price = day_data["pre_market_data"].loc[str(dia)+' 09:29:00-04:00']['High']
            except KeyError:
                pre_high_price = 'NaN'
            
            try:
                pre_low_price = day_data["pre_market_data"].loc[str(dia)+' 09:29:00-04:00']['Low']
            except KeyError:
                pre_low_price = 'NaN'
        
            # Append the row to the list
            rows.append({
            'Symbol': 'NaN',
            'Day': self.get_day_of_week(dia),
            'Date': datetime.strptime(dia, '%Y-%m-%d').strftime('%d-%m-%Y'),
            'Close 8:29am': pre_close_price,
            'Open 9:30am': 'NaN',
            'Opening Gap': 'NaN',
            'Volume 9:30am': 'NaN',
            'High 8:29am': pre_high_price,
            'Low 8:29am': pre_low_price,
            'High 9:30am': 'NaN',
            'Low 9:30am': 'NaN'
            })

        # Iterate over the dates and extract the required details
        for day_data in data:
            if current==True:
                const = 1
            else:
                const = 0
            for j in range(len(dates)-const):
            
                dia = dates[j]
                if day_data["date"] == dia:
                    try:
                        # Access the pre-market data for the specific time
                        pre_close_price = day_data["pre_market_data"].loc[str(dia)+' 09:29:00-04:00']['Close']
                    except KeyError:
                        pre_close_price = 'NaN'
                    
                    try:
                        pre_high_price = day_data["pre_market_data"].loc[str(dia)+' 09:29:00-04:00']['High']
                    except KeyError:
                        pre_high_price = 'NaN'
                    
                    try:
                        pre_low_price = day_data["pre_market_data"].loc[str(dia)+' 09:29:00-04:00']['Low']
                    except KeyError:
                        pre_low_price = 'NaN'

                    try:
                        # Access the open market data for the specific time
                        post_open_price = day_data["open_market_data"].loc[str(dia)+' 09:30:00-04:00']['Open']
                    except KeyError:
                        post_open_price = 'NaN'
                    
                    try:
                        post_volume = day_data["open_market_data"].loc[str(dia)+' 09:30:00-04:00']['Volume']
                    except KeyError:
                        post_volume = 'NaN'
                    
                    try:
                        post_high_price = day_data["open_market_data"].loc[str(dia)+' 09:30:00-04:00']['High']
                    except KeyError:
                        post_high_price = 'NaN'
                    
                    try:
                        post_low_price = day_data["open_market_data"].loc[str(dia)+' 09:30:00-04:00']['Low']
                    except KeyError:
                        post_low_price = 'NaN'

                    # Calculate the gap between the pre-market close and open market open
                    if pre_close_price != 'NaN' and post_open_price != 'NaN':
                        gap = post_open_price - pre_close_price
                    else:
                        gap = 'NaN'


                    # Append the row to the list
                    rows.append({
                    'Symbol': symbol[j],
                    'Day': self.get_day_of_week(dia),
                    'Date': datetime.strptime(dia, '%Y-%m-%d').strftime('%d-%m-%Y'),
                    'Close 8:29am': pre_close_price,
                    'Open 9:30am': post_open_price,
                    'Opening Gap': gap,
                    'Volume 9:30am': post_volume,
                    'High 8:29am': pre_high_price,
                    'Low 8:29am': pre_low_price,
                    'High 9:30am': post_high_price,
                    'Low 9:30am': post_low_price
                    })

        # Create a DataFrame from the list of rows
        df = pd.DataFrame(rows)

        # Highlight the date
        df.style.applymap(lambda value: self.highlight_col(value, dates), subset=['Date'])

        return df

