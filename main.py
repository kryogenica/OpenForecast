import streamlit as st
from stock_plotter import StockPlotter
from stock_collectors import stockChecker
from stock_analyzers import stockAnalyzer, stockPredictor, stockNormalizer
from datetime import datetime, timedelta
import pytz
import time
import streamlit.components.v1 as components


st.set_page_config(
    page_title="My App",
    layout="wide",
    menu_items={
        'Get Help': 'https://www.example.com',
        'Report a bug': "https://www.example.com",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
# Instantiate the StockPlotter and other classes
SC = stockChecker()
SA = stockAnalyzer()
SN = stockNormalizer()
stock_plotter = StockPlotter(SC, SA, SN)
# ============
#    Current and most current trading day.
# ============
latest_day = datetime.now().strftime('%Y-%m-%d')
while not SC.is_it_a_trading_day(latest_day):
    latest_day = (datetime.strptime(latest_day, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')

# Get the current time in NYC
nyc_tz = pytz.timezone('America/New_York')
current_time_nyc = datetime.now(nyc_tz).time()

# Define the market open and close times
market_open_time = datetime.strptime("09:30", "%H:%M").time()
market_close_time = datetime.strptime("23:00", "%H:%M").time()

active_trading = False
if latest_day == datetime.now().strftime('%Y-%m-%d'):
    if market_open_time <= current_time_nyc <= market_close_time:
        active_trading = True
        today = datetime.now().strftime('%Y-%m-%d')
        latest_day = (datetime.strptime(latest_day, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')

# ========================
#    Streamlit Initialization
# ========================

# Initialize session state for the buttons
if 'active_feature' not in st.session_state:
    
    st.session_state['active_feature'] = 'Most Recent'
    st.session_state['T_or_F_exists'] = False
    st.session_state['T_or_F_with_data'] = True
    st.session_state['exchange_name'] = []
    st.session_state['stock'] = None
    st.session_state['latest_day_stock_data'] = None
    st.session_state['active_stock_data'] = None
    st.session_state['lists_of_measures'] = None
    st.session_state['predictor_option'] = 'DTW Regresion'
    metric_option = 'Pearson Correlation'
    st.session_state['smoothing_window'] = 5
    st.session_state['Trading_day'] = latest_day
    st.session_state['input_label'] = "Type Stock Indicator here:"
    st.session_state['matching_window'] = -30

# ========================
#    Streamlit Sidebar
# ========================

# Create a sidebar and add mutually exclusive buttons
with st.sidebar:
    # Create two columns in the sidebar for the buttons
    col1, col2 = st.columns(2)

    # Logic to display Feature: "Most Recent" or Feature 2: "Currently Trading" mutually exclusive
    with col1:
        if st.button("Most Recent"):
            st.session_state['active_feature'] = 'Most Recent'
            st.session_state['input_label'] = "Type Stock Indicator here:"

    with col2:
        if active_trading:
            if st.button("Open Trading"):
                st.session_state['active_feature'] = 'Currently Trading'
                st.session_state['input_label'] = "Type Stock Indicator here:"

        else:
            st.markdown(
            """
            <style>
            .tooltip {
                position: relative;
                display: inline-block;
                cursor: not-allowed;
            }

            .tooltip .tooltiptext {
                visibility: hidden;
                width: 120px;
                background-color: black;
                color: #fff;
                text-align: center;
                border-radius: 16px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                top: 110%; /* Position the tooltip below the button */
                left: 50%;
                margin-left: -60px;
                opacity: 0;
                transition: opacity 0.3s;
            }

            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }

            .tooltip button {
                border-radius: 100px; /* Make the button more round */
            }
            </style>
            <div class="tooltip">
                <button disabled style="cursor: not-allowed;">Open Trading</button>
                <span class="tooltiptext">NYSE & Nasdaq are closed</span>
            </div>
            """,
            unsafe_allow_html=True
            )

    if st.session_state['active_feature'] == 'Most Recent':
        st.markdown("<p style='font-size:13px;'>You are now viewing the two-hour period immediately after the market opened on the last trading day.</p>", unsafe_allow_html=True)
        st.session_state['Trading_day'] = latest_day
        print(st.session_state['Trading_day'])
        
    elif st.session_state['active_feature'] == 'Currently Trading':
        # Custom CSS for neon glow effect
        stock_plotter.display_neon_text(text="Market is Open", font_size=20, color="blue")
        st.markdown("<p style='font-size:13px;'>You are now viewing the current period immediately after the market opened on todays trading day.</p>", unsafe_allow_html=True)
        st.session_state['Trading_day'] = datetime.now().strftime('%Y-%m-%d')
        print(st.session_state['Trading_day'])
        

    # User input
    user_input = st.text_input(st.session_state['input_label'], "")
    if user_input:
        # Store user input in session state
        st.session_state['user_input'] = user_input
        
        # Check if the stock is valid
        st.session_state['T_or_F_exists'] = SC.is_valid_stock(st.session_state['user_input'])
        
        # Store the stock symbol in session state
        st.session_state['stock'] = str(st.session_state['user_input'])
        
        if st.session_state['T_or_F_exists']:
            # Check if the stock has data and get the exchange name
            st.session_state['T_or_F_with_data'], st.session_state['exchange_name'] = SC.stock_has_data(st.session_state['user_input'])
            
            if st.session_state['T_or_F_with_data']:
                # Update the input label to indicate the stock is identified
                st.session_state['input_label'] = (f"✅ {st.session_state['stock'].upper()} stock identified!")
                
                # Get the stock data for the latest trading day
                st.session_state['latest_day_stock_data'] = SC.get_last_trading_days(st.session_state['stock'].upper(), latest_day)
                
                if active_trading:
                    # If the market is currently open, get the stock data for today
                    st.session_state['active_stock_data'] = SC.get_last_trading_days(st.session_state['stock'].upper(), datetime.now().strftime('%Y-%m-%d'), back_window=1)
                    # Append the latest day stock data to the active stock data
                    st.session_state['active_stock_data'].extend(st.session_state['latest_day_stock_data'])
                                
                    print("Data has been collected.")
                    # Update the label in the sidebar and re-run the app
                    user_input = ""
                    st.rerun()
            else:
                st.session_state['input_label'] = (f"❓ Not enough data for {st.session_state['stock'].upper()} in a {st.session_state['exchange_name']} exchange!")
                st.session_state['T_or_F_exists'] = False
                st.rerun()
        else:         
            st.session_state['input_label'] = (f"❌ {st.session_state['stock'].upper()} stock not found!")
            st.session_state['T_or_F_exists'] = False
            st.rerun()

        

    # Create a radio button selection in the sidebar
    metric_option = st.sidebar.radio(
        "Choose the type of similarity metric:",
        ('Pearson Correlation', 'Kendall Tau', 'Dynamic Time Warping', 'Mutual Information')
    )
    measures_to_num_n_type = {'Pearson Correlation': [0,'Max'], 'Kendall Tau': [1,'Max'], 'Dynamic Time Warping': [2,'Min'], 'Mutual Information': [3,'Max']}

    # Add a numeric scroller (slider) for controlling a window used for prediction
    st.session_state['matching_window'] = st.sidebar.slider(
        "Similarity metric window [min] applied on pre-open markets:",
        min_value=-210, 
        max_value=-30, 
        value=-30,  # Default value
        step=30
    )

    st.write("")

    # Create a radio button selection in the sidebar
    st.session_state['predictor_option'] = st.sidebar.radio(
        "Choose the type of prediction mechanism:",
        ('DTW Regresion', 'Ridge regression', 'Elastic Net')
    )

    # Add a numeric scroller (slider) for controlling a window used for prediction
    prediction_vision = st.sidebar.slider(
        "Window used for prediction [min]:",
        min_value=1, 
        max_value=180, 
        value=5,  # Default value
        step=1
    )

    # Add a numeric scroller (slider) for controlling a parameter
    new_smoothing_window = st.sidebar.slider(
        "Rolling mean window [min]:",
        min_value=1, 
        max_value=10, 
        value=3,  # Default value
        step=1
    )

    # Check if the smoothing window has changed
    if new_smoothing_window != st.session_state['smoothing_window']:
        st.session_state['smoothing_window'] = new_smoothing_window
    
    # Display the current time in NYC in hours, minutes, and seconds
    time_placeholder = st.empty()
    current_time_nyc = datetime.now(nyc_tz).strftime("%H:%M:%S")
    time_placeholder.write(f"Last refreshed at {current_time_nyc} NYC time")

# ========================
#    Streamlit Block-container
# ========================

# JavaScript code to maintain the scroll position
js_code = """
    <script>
    // Check if scroll position is already set in sessionStorage, if not, set it to 0
    if (!sessionStorage.getItem('scrollpos')) {
        sessionStorage.setItem('scrollpos', 0);
    }

    window.addEventListener('DOMContentLoaded', (event) => {
        // Get the current scroll position
        let scrollPos = sessionStorage.getItem('scrollpos');
        if (scrollPos) window.scrollTo(0, scrollPos);
    });

    window.onscroll = function(e) {
        // Store the current scroll position in sessionStorage
        sessionStorage.setItem('scrollpos', window.scrollY);
    };
    </script>
"""

# Title for the app
st.title("Open Forecast")

# Create a container for the main content
with st.container():
    # Check if stock data exists and call the plot function
    if st.session_state['T_or_F_exists']:
        
        if st.session_state['active_feature'] == 'Currently Trading':
            st.session_state['stock_data'] = st.session_state['active_stock_data']

        elif st.session_state['active_feature'] == 'Most Recent':
            st.session_state['stock_data'] = st.session_state['latest_day_stock_data']


        stock_data_last_days = st.session_state['stock_data']
        st.session_state['lists_of_measures'] = SA.get_pre_market_measures(st.session_state['stock_data'], st.session_state['smoothing_window'], st.session_state['matching_window'])
        best_pre_market_match = st.session_state['lists_of_measures'][measures_to_num_n_type[metric_option][0]]
        circle_indexes = [i for i, _ in SA.max_min_of_abs(best_pre_market_match, measures_to_num_n_type[metric_option][1])]  # Get top 3 best results
        best_open_data_indexes = [i + 1 for i in circle_indexes]

        # Get the dates from the stock data using the best open data indexes
        best_open_data_dates = [st.session_state['stock_data'][i]['date'] for i in best_open_data_indexes]
        best_open_data_dates = best_open_data_dates + [st.session_state['stock_data'][0]['date']]

        exogenous_series = [SN.normalize_market_data(st.session_state['stock_data'][i], st.session_state['smoothing_window'])[1] for i in best_open_data_indexes]
        endogenous_series = SN.normalize_market_data(st.session_state['stock_data'][0], 1)[1]

        prediction_machine = stockPredictor(exogenous_series, endogenous_series)
        prediction_machine.data_divider(prediction_vision)
        if st.session_state['predictor_option'] == 'DTW Regresion':
            prediction = prediction_machine.DTW_regresion()
        elif st.session_state['predictor_option'] == 'Ridge regression':
            prediction = prediction_machine.ridge_model()
        elif st.session_state['predictor_option'] == 'Elastic Net':
            prediction = prediction_machine.elastic_net()

        if 'Trading_day' in st.session_state and st.session_state['Trading_day'] == datetime.now().strftime('%Y-%m-%d'):
            special_case = True
        else:
            special_case = False
            
        stock_plotter.plot_pre_n_open_market_interactive(st.session_state['stock_data'], str(st.session_state['stock']), best_open_data_indexes, prediction_vision, st.session_state['matching_window'], best_open_data_dates, st.session_state['smoothing_window'], special_case, prediction)  # Plot the data after user input
    else:
        # Display an animated GIF
        st.write("Waiting for user input...")
        st.image("https://media.giphy.com/media/6oeRBKg7mwEZnSnYkn/giphy.gif", use_column_width=True)

    # Check if stock data exists and call the plot function
    if st.session_state['T_or_F_exists']:
        if st.session_state['active_feature'] == 'Currently Trading':
            st.session_state['stock_data'] = st.session_state['active_stock_data']
            
        elif st.session_state['active_feature'] == 'Most Recent':
            st.session_state['stock_data'] = st.session_state['latest_day_stock_data']

        st.session_state['lists_of_measures'] = SA.get_pre_market_measures(st.session_state['stock_data'], st.session_state['smoothing_window'], st.session_state['matching_window'])
        best_pre_market_match = st.session_state['lists_of_measures'][measures_to_num_n_type[metric_option][0]]
        circle_indexes = [i for i, _ in SA.max_min_of_abs(best_pre_market_match, measures_to_num_n_type[metric_option][1])]  # Get top 3 best results
        dates_to_display = [st.session_state['stock_data'][i]['date'] for i in range(len(st.session_state['stock_data']))]
        stock_plotter.plot_horizontal_heatmap(best_pre_market_match, circle_indexes, dates_to_display)  # Plot the data after user input

    # Check if stock data exists and call the plot function
    if st.session_state['T_or_F_exists']:
        if st.session_state['active_feature'] == 'Currently Trading':
            st.session_state['stock_data'] = st.session_state['active_stock_data']
            
        elif st.session_state['active_feature'] == 'Most Recent':
            st.session_state['stock_data'] = st.session_state['latest_day_stock_data']
        stock_plotter.display_stock_details(st.session_state['stock_data'], best_open_data_dates)

    # Display the JavaScript in the Streamlit app
    components.html(js_code)

# ========================