import streamlit as st
from stock_collectors import StockChecker
import matplotlib.pyplot as plt

SC = StockChecker()
def get_trading_data(Ticker):
    prices = SC.get_last_trading_days(Ticker.upper(), '2024-09-09')
    return prices

def plot_pre_market(stock_data_last_days):
    plt.figure(figsize=(10,6))
    # Calculate and plot rolling mean for each day's open market data
    for i, day_data in enumerate(stock_data_last_days):

        delta = 0.5*((i+1)/len(stock_data_last_days))
        rolling_mean = day_data["open_market_data"]["Close"].rolling(window=10).mean()
        normalizer = rolling_mean.iloc[9]
        plt.plot(range(0,len(rolling_mean)),  ((rolling_mean-normalizer)/normalizer)*100, color=(0.5+delta,0.5+delta,(delta)*1.8))

    # Add labels and title
    plt.xlabel("Time")
    plt.ylabel("Rolling Mean (Close Price)")
    plt.title("Rolling Mean for Open Market Data")
    # Show the plot within the Streamlit app
    st.pyplot(plt)

# Title for the app
st.title("Open Forecat")

# Initialize session state for the buttons
if 'active_feature' not in st.session_state:
    st.session_state['active_feature'] = 'feature_1'


# Create a sidebar and add mutually exclusive buttons
with st.sidebar:
    st.write("## Controls")

    # Create two columns in the sidebar for the buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Feature 1"):
            st.session_state['active_feature'] = 'feature_1'

    with col2:
        if st.button("Feature 2"):
            st.session_state['active_feature'] = 'feature_2'
    
    # A line for separation
    # Logic to display Feature 1 or Feature 2, mutually exclusive
    if st.session_state['active_feature'] == 'feature_1':
        st.write("Feature 1 is now visible!")
    else:
        st.write("Feature 2 is now visible!")
    
    # Initialize session state for the label if it's not set
    if 'input_label' not in st.session_state:
        st.session_state['input_label'] = "Type Stock Indicator here:"

    # User input
    user_input = st.text_input(st.session_state['input_label'], "")
    if user_input:
        T_or_F = SC.is_valid_stock(user_input)

        if T_or_F:
            st.session_state['input_label'] = (f"✅ {user_input} stock indetified!")
            stock_data_last_days = get_trading_data(user_input)
            print("Data has been collected.")
            st.session_state['stock_data'] = stock_data_last_days  # Store the data in session state

        else:         
            st.session_state['input_label'] = (f"❌ {user_input} stock not found!")
        # Clear the input box after processing
        st.rerun()



# Check if stock data exists and call the plot function
if 'stock_data' in st.session_state:
    stock_data_last_days = st.session_state['stock_data']
    plot_pre_market(stock_data_last_days)  # Plot the data after user input
