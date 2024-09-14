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
        full_day_data_5m = stock.history(start=day, end=(datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"), interval="1m", prepost=True)

        # Convert the index to a datetime index for filtering
        full_day_data_1m.index = pd.to_datetime(full_day_data_1m.index)
        full_day_data_5m.index = pd.to_datetime(full_day_data_5m.index)

        # Filter pre-market data (4:00 AM to 9:30 AM) at 1-minute intervals
        pre_market_data = full_day_data_5m.between_time("06:01", "09:30")

        # Filter open market data (9:30 AM to 10:30 AM) at 1-minute intervals
        open_market_data = full_day_data_1m.between_time("09:30", "11:30")

        return {
            "pre_market_data": pre_market_data,
            "open_market_data": open_market_data
        }
    
    def get_last_trading_days(self, ticker: str, date, back_window=25):
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