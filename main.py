import streamlit as st
from stock_collectors import stockChecker
from stock_analyzers import stockAnalyzer
import matplotlib.pyplot as plt
import numpy as np

SC = stockChecker()
SA = stockAnalyzer()
# ============
#    Functions
# ============
def get_trading_data(Ticker):
    prices = SC.get_last_trading_days(Ticker.upper(), '2024-09-09')
    return prices

def plot_pre_n_open_market(stock_data_last_days, ticker_name, best_pre_open_indicators, prediction=None):

    best_pre_open_data = [i+1 for i in best_pre_open_indicators]
    # Calculate and plot rolling mean for each day's open market data
    for i, day_data in enumerate(stock_data_last_days):
        if i not in best_pre_open_data:
            delta = 0.5*((i+1)/len(stock_data_last_days))
            data = day_data["pre_market_data"]["Close"]
            pre_data = data.rolling(window=5).mean()
            pre_normalizer = data.iloc[-1]

            data = day_data["open_market_data"]["Open"]
            open_data = data.rolling(window=5).mean().shift(-2)
            open_normalizer = data.iloc[0]

            plt.plot(range(-len(pre_data), 0),  ((pre_data-pre_normalizer)/pre_normalizer)*100, color=(0.45+delta,0.5+delta,(delta)*1.8), linewidth=0.4)
            plt.plot(range(0, len(open_data)),  ((open_data-open_normalizer)/open_normalizer)*100, color=(0.45+delta,0.5+delta,(delta)*1.8), linewidth=0.4)

    k = 0
    colors = ['red','blue','violet']
    for i, day_data in enumerate(stock_data_last_days):
        if i in best_pre_open_data:
            data = day_data["pre_market_data"]["Close"]
            pre_data = data.rolling(window=5).mean()
            pre_normalizer = data.iloc[-1]

            data = day_data["open_market_data"]["Open"]
            open_data = data.rolling(window=5).mean().shift(-2)
            open_normalizer = data.iloc[0]
            
            plt.plot(range(-len(pre_data),0),  ((pre_data-pre_normalizer)/pre_normalizer)*100, color=colors[k], linewidth=2)
            plt.plot(range(0,len(open_data)),  ((open_data-open_normalizer)/open_normalizer)*100, color=colors[k], linewidth=2)
            k += 1

    data = stock_data_last_days[0]["pre_market_data"]["Close"]
    pre_data = data.rolling(window=5).mean()
    pre_normalizer = data.iloc[-1]

    data = stock_data_last_days[0]["open_market_data"]["Open"]
    open_data = data.rolling(window=5).mean().shift(-2)
    open_normalizer = data.iloc[0]
    plt.plot(range(-len(pre_data),0),  ((pre_data-pre_normalizer)/pre_normalizer)*100, color='black', linewidth=3, linestyle=':')
    plt.plot(range(0,len(open_data)),  ((open_data-open_normalizer)/open_normalizer)*100, color='black', linewidth=3, linestyle=':')

    plt.axhline(y=0, linewidth=0.5, color='black')
    plt.axvline(x=0, linewidth=0.5, color='black')

    if prediction != None:
        plt.plot(range(0,len(prediction)), prediction, color='black',  linewidth=3)

    # Add labels and title
    plt.xlabel("Minutes before and after Opening")
    plt.ylabel("Close Price % | Open Price %")
    plt.title(f"Rolling Mean for {ticker_name.upper()} Market Data")
    plt.show()
    # Show the plot within the Streamlit app
    st.pyplot(plt)

def plot_horizontal_heatmap(values, circle_indexes):
  """
  Plots a list of values as a vertical heatmap.
  Args:
    values: A list of values to plot.
  """
  # Create the plot
  fig, ax = plt.subplots(figsize=(6,1), dpi=100)

  # Create the heatmap
  im = ax.imshow([values], cmap='viridis', aspect='auto')

  # Add value annotations
  for i in range(len(values)):
    ax.text(i, 0, f"{values[i]:.2f}", ha="center", va="center", color="brown", fontsize=60, rotation=70)

  # Add circles to specified indexes
  colors = ['blue','red','violet']
  for j, i in enumerate(circle_indexes):
    ax.plot(i, -0.4, marker='o', markersize=50, color=colors[j])
    ax.plot(i, 0.4, marker='o', markersize=50, color=colors[j])

  # Remove colorbar
  ax.get_yaxis().set_visible(False)
  ax.get_xaxis().set_visible(False)
  ax.set_frame_on(False)

  # Remove the white edges by adjusting the layout
  plt.subplots_adjust(left=0, right=5, top=5, bottom=0)

  # Display the plot in Streamlit without white edges
  st.pyplot(fig)
# ============
#    Streamlit
# ============
# Title for the app
st.title("Open Forecat")

# Initialize session state for the buttons
if 'active_feature' not in st.session_state:
    st.session_state['active_feature'] = 'feature_1'
    st.session_state['T_or_F_exists'] = False
    st.session_state['T_or_F_with_data'] = True
    st.session_state['exchange_name'] = []
    st.session_state['stock'] = None
    option = 'Pearson Correlation'


# Create a sidebar and add mutually exclusive buttons
with st.sidebar:
    st.write("## Controls")

    # Create two columns in the sidebar for the buttons
    col1, col2 = st.columns(2)

    # Logic to display Feature 1 or Feature 2, mutually exclusive
    with col1:
        if st.button("Feature 1"):
            st.session_state['active_feature'] = 'feature_1'

    with col2:
        if st.button("Feature 2"):
            st.session_state['active_feature'] = 'feature_2'
    
    if st.session_state['active_feature'] == 'feature_1':
        st.write("Feature 1 is now enabled!")
    else:
        st.write("Feature 2 is now enabled!")
    
    # Initialize session state for the label if it's not set
    if 'input_label' not in st.session_state:
        st.session_state['input_label'] = "Type Stock Indicator here:"

    # User input
    user_input = st.text_input(st.session_state['input_label'], "")
    if user_input:

        st.session_state['T_or_F_exists'] = SC.is_valid_stock(user_input)
        st.session_state['stock'] = str(user_input)
        if st.session_state['T_or_F_exists'] :
            st.session_state['T_or_F_with_data'], st.session_state['exchange_name'] = SC.stock_has_data(user_input)
            if st.session_state['T_or_F_with_data']:
                st.session_state['input_label'] = (f"✅ {st.session_state['stock'].upper()} stock indetified!")
                stock_data_last_days = get_trading_data(st.session_state['stock'])
                print("Data has been collected.")
                st.session_state['stock_data'] = stock_data_last_days  # Store the data in session state
                st.session_state['lists_of_measures'] = SA.get_pre_market_measures(stock_data_last_days)
            else:
                st.session_state['input_label'] = (f"❓ Not enough data for {st.session_state['stock'].upper()} in a {st.session_state['exchange_name']} exchange!")
                st.session_state['T_or_F_exists'] = False
        else:         
            st.session_state['input_label'] = (f"❌ {st.session_state['stock'].upper()} stock not found!")
        # Clear the input box after processing
        st.rerun()

    # Create a radio button selection in the sidebar
    option = st.sidebar.radio(
        "Choose the type of measurement:",
        ('Pearson Correlation', 'Kendall Tau', 'Dynamic Time Warping', 'Mutual Information')
    )
    measures_to_num_n_type = {'Pearson Correlation': [0,'Max'], 'Kendall Tau': [1,'Max'], 'Dynamic Time Warping': [2,'Min'], 'Mutual Information': [3,'Max']}



# Check if stock data exists and call the plot function
if st.session_state['T_or_F_exists']:
    stock_data_last_days = st.session_state['stock_data']
    best_pre_market_match = st.session_state['lists_of_measures'][measures_to_num_n_type[option][0]]
    circle_indexes = [i for i, _ in SA.max_min_of_abs(best_pre_market_match, measures_to_num_n_type[option][1])]# Get top 3 best results
    plot_pre_n_open_market(stock_data_last_days, str(st.session_state['stock']), circle_indexes)  # Plot the data after user input
else:
    # Display an animated GIF
    st.image("https://media.giphy.com/media/6oeRBKg7mwEZnSnYkn/giphy.gif")


# Check if stock data exists and call the plot function
if st.session_state['T_or_F_exists']:
    best_pre_market_match = st.session_state['lists_of_measures'][measures_to_num_n_type[option][0]]
    circle_indexes = [i for i, _ in SA.max_min_of_abs(best_pre_market_match, measures_to_num_n_type[option][1])]# Get top 3 best results
    plot_horizontal_heatmap(best_pre_market_match, circle_indexes)  # Plot the data after user input
else:
    # Display an animated GIF
    st.image("https://media.giphy.com/media/6oeRBKg7mwEZnSnYkn/giphy.gif", caption="Waiting for user input...")


