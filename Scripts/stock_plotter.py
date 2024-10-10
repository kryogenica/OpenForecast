import streamlit as st
from datetime import datetime
from matplotlib.colors import ListedColormap
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt




class StockPlotter:
    def __init__(_self, stock_checker, stock_analyzer, stock_normalizer):
        _self.SC = stock_checker
        _self.SA = stock_analyzer
        _self.SN = stock_normalizer

    def remove_trailing_duplicates(self, data):
            last_value = data[-1]
            for i in range(len(data) - 1, -1, -1):
                if data[i] != last_value:
                    return data[:i + 1]

    @st.cache_data
    def plot_pre_n_open_market_interactive(_self, stock_data_last_days, ticker_name, best_open_data_indexes, vision, matching_window, dates, smoothing_window, special_case, prediction=None):
        window = smoothing_window
        all_values = []
        pre_lengths = []
        open_lengths = []
        # Create a plotly figure
        fig = go.Figure()

        # Calculate and plot rolling mean for each day's open market data
        for i, day_data in enumerate(stock_data_last_days):
            if i not in best_open_data_indexes and i > 0:
                delta = 0.5 * ((i + 1) / len(stock_data_last_days))
                normalized_pre_data, normalized_open_data = _self.SN.normalize_market_data(day_data, window)
                pre_lengths.append(len(normalized_pre_data))
                open_lengths.append(len(normalized_open_data))
                all_values.extend(normalized_pre_data)
                all_values.extend(normalized_open_data)
                fig.add_trace(go.Scatter(
                    x=list(range(-len(normalized_pre_data), 0)),
                    y=normalized_pre_data,
                    mode='lines',
                    line=dict(color=f'rgba({int((0.45 + delta) * 255)}, {int((0.5 + delta) * 255)}, {int(delta * 1.8 * 255)}, 0.5)', width=0.3),
                    name=f'Day {i} Pre-market',
                    hoverinfo='skip'  # Make this trace non-interactive
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(0, len(normalized_open_data))),
                    y=normalized_open_data,
                    mode='lines',
                    line=dict(color=f'rgba({int((0.45 + delta) * 255)}, {int((0.5 + delta) * 255)}, {int(delta * 1.8 * 255)}, 0.5)', width=0.3),
                    name=f'Day {i} Open-market',
                    hoverinfo='skip'  # Make this trace non-interactive
                ))

        

        k = 0
        colors = ['blue', 'red', 'violet']
        symbol = ['▲', '⬟', '■']
        names_of_lines = []
        # Plot selected pre and open market data with most similar pre-open market data to latest day
        for i, day_data in enumerate(stock_data_last_days):
            if i in best_open_data_indexes:
                normalized_pre_data, normalized_open_data = _self.SN.normalize_market_data(day_data, window)
                pre_lengths.append(len(normalized_pre_data))
                open_lengths.append(len(normalized_open_data))
                all_values.extend(normalized_pre_data)
                all_values.extend(normalized_open_data)
                fig.add_trace(go.Scatter(
                    x=list(range(-len(normalized_pre_data), 0)),
                    y=normalized_pre_data,
                    mode='lines',
                    line=dict(color=colors[k], width=2, dash='dot'),
                    name=f'Best Day {i} Pre-market',
                    hovertemplate=f'Compatible trading data<br>Date: {dates[k]}<br>Time: %{{x}} min<br>Close: %{{y:.2f}}%<extra></extra>'  # Format hover to 2 decimal places
                ))
                names_of_lines.append(f'Matching Close-market data {symbol[k]}')
                fig.add_trace(go.Scatter(
                    x=list(range(0, len(normalized_open_data))),
                    y=normalized_open_data,
                    mode='lines',
                    line=dict(color=colors[k], width=2, dash='dot'),
                    name=f'Matching Close-market data {symbol[k]}',
                    hovertemplate=f'Compatible trading data<br>Date: {dates[k]}<br>Time: %{{x}} min<br>Open: %{{y:.2f}}%<extra></extra>'  # Format hover to 2 decimal places
                ))
                k += 1

        # Plot selected pre and open market data of latest day
        normalized_pre_data, _ = _self.SN.normalize_market_data(stock_data_last_days[0], window, special_case=special_case)
        _, normalized_open_data = _self.SN.normalize_market_data(stock_data_last_days[0], 1, special_case=special_case)
        pre_lengths.append(len(normalized_pre_data))
        open_lengths.append(len(normalized_open_data))
        all_values.extend(normalized_pre_data)
        all_values.extend(normalized_open_data)
        fig.add_trace(go.Scatter(
            x=list(range(-len(normalized_pre_data), 0)),
            y=normalized_pre_data,
            mode='lines',
            line=dict(color='gray', width=4),
            name='Latest Day Pre-market',
            hovertemplate=f'Most recent trading data<br>Date: {dates[-1]}<br>Time: %{{x}} min<br>Close: %{{y:.2f}}%<extra></extra>'  # Format hover to 2 decimal places
        ))
        fig.add_trace(go.Scatter(
            x=list(range(0, len(normalized_open_data))),
            y=normalized_open_data,
            mode='lines',
            line=dict(color='gray', width=4),
            name='Latest Open-market data',
            hovertemplate=f'Most recent trading data<br>Date: {dates[-1]}<br>Time: %{{x}} min<br>Open: %{{y:.2f}}%<extra></extra>'  # Format hover to 2 decimal places
        ))

        # Set y-axis limits based on the collected values before plotting the prediction
        y_min, y_max = min(all_values), max(all_values)
        fig.update_yaxes(range=[y_min, y_max])

        if prediction is not None and prediction.any():
            fig.add_trace(go.Scatter(
                x=list(range(0, len(prediction))),
                y=prediction,
                mode='lines',
                line=dict(color='gold', width=2),
                name='Prediction',
                hovertemplate='Prediction<br>Time: %{x} min<br>Open: %{y:.2f} %<extra></extra>'  # Format hover to 2 decimal places
            ))

        # Calculate tick values for every 30 minutes interval
        negative_tick_vals = list(range(0, max(pre_lengths),  30))
        negative_tick_vals.reverse()
        negative_tick_vals = [-1 * int(i) for i in negative_tick_vals]
        positive_tick_vals = list(range(0, max(open_lengths), 30))
        tick_vals = negative_tick_vals + positive_tick_vals
        tick_text = [f'{t}' for t in tick_vals]  # Label each tick

        # Add vertical lines at each tick mark
        for tick in tick_vals:
            fig.add_vline(
                x=tick,
                line=dict(color='gray', width=0.2, dash='dash'),
                layer='below'  # Put the lines below the traces
            )

        # Add vertical and horizontal lines
        fig.add_shape(type='line', x0=0, y0=y_min, x1=0, y1=y_max,
                      line=dict(color='white', width=0.5))
        fig.add_shape(type='line', x0=-max(pre_lengths), y0=0, x1=max(open_lengths), y1=0,
                      line=dict(color='white', width=0.5))

        # Add shading
        fig.add_vrect(x0=0, x1=vision, fillcolor='pink', opacity=0.2, line_width=0)
        fig.add_vrect(x0=matching_window, x1=0, fillcolor='blue', opacity=0.1, line_width=0)

        # Set layout
        fig.update_layout(
            title=f"Rolling Mean for {ticker_name.upper()} Market Data",
            xaxis_title="Minutes before and after Opening",
            yaxis_title="Close Price % | Open Price %",
            xaxis=dict(
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text
            ),
            showlegend=True,
            legend=dict(
            itemsizing='constant',
            traceorder='normal',
            itemclick=False,
            itemdoubleclick=False,
            x=0.01,  # Position the legend inside the graph
            y=0.99,  # Position the legend inside the graph
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.1)'  # Semi-transparent background for better readability
            )
        )
        names_of_lines.append('Prediction')
        names_of_lines.append('Latest Open-market data')
        # Update traces to show legend only for 'Prediction', 'Latest Open-market data', and 'Best matching Open-market data'
        for trace in fig.data:
            if trace.name not in names_of_lines:
                trace.showlegend = False


        # Display the figure in Streamlit
        st.plotly_chart(fig)

    @st.cache_data
    def plot_horizontal_heatmap(_self, values, circle_indexes, dates_to_display):
        
        """
        Plots a list of values as a vertical heatmap.
        Args:
            values: A list of values to plot.
        """
        # Calculate the difference in days between the first date and the rest
        first_date = datetime.strptime(dates_to_display[0], '%Y-%m-%d')
        day_difference = [(first_date - datetime.strptime(dates_to_display[i], '%Y-%m-%d') ).days for i in range(1, len(dates_to_display))]

        # Create the plot
        fig, ax = plt.subplots(figsize=(5,0.4), dpi=200)

        # Create a colormap that maps NaN values to black

        # Create a colormap that maps NaN values to black
        cmap = plt.cm.viridis
        cmap.set_bad(color='black')

        # Convert the values to a numpy array and mask NaNs
        values_array = np.array(values)
        masked_values = np.ma.masked_invalid(values_array)

        # Create the heatmap with the masked array
        im = ax.imshow([masked_values], cmap=cmap, aspect='auto')

        # Add circles to specified indexes
        colors = ['blue','red','violet']
        symbols = ['^','p','s']
        for j, i in enumerate(circle_indexes):
            ax.plot(i, 0, marker=symbols[j], markersize=60, color=colors[j])


        # Add value annotations
        for i in range(len(values)):
            if values[i] >= 10:
                ax.text(i, 0.35, f"{int(values[i])}", ha="center", va="center", color="black", fontsize=30)
            else:
                ax.text(i, 0.35, f"{values[i]:.2f}", ha="center", va="center", color="black", fontsize=30)
            if i == len(values) - 1:
                ax.text(i, -0.13, f"{day_difference[i]}\ndays\nago", ha="center", va="center", color="white", fontsize=30, fontweight='bold')
            else:
                ax.text(i, -0.35, f"{day_difference[i]}", ha="center", va="center", color="white", fontsize=30, fontweight='bold')


        # Remove colorbar
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_frame_on(False)

        # Remove the white edges by adjusting the layout
        plt.subplots_adjust(left=0, right=5, top=5, bottom=0)

        # Display the plot in Streamlit without white edges
        st.pyplot(fig)

    @st.cache_data
    def display_stock_details(_self, stock_data, best_open_data_dates):
        df = _self.SC.get_details_on_stock_per_date(stock_data, best_open_data_dates, False)  # Plot the data after user input
        st.markdown(
                """
                <style>
                .stDataFrame div[data-testid="stHorizontalBlock"] {
                overflow: visible;
                width: 100%;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        st.dataframe(df, use_container_width=True)


    def display_neon_text(self, text, font_size, color):
        neon_style = f"""
            <style>
            @keyframes neon-glow {{
                0% {{
                    text-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 20px #fff, 0 0 40px {color}, 0 0 80px {color}, 0 0 90px {color}, 0 0 100px {color}, 0 0 150px {color};
                }}
                50% {{
                    text-shadow: 0 0 3px #fff, 0 0 5px #fff, 0 0 10px #fff, 0 0 20px {color}, 0 0 30px {color}, 0 0 35px {color}, 0 0 40px {color}, 0 0 50px {color};
                }}
                100% {{
                    text-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 20px #fff, 0 0 40px {color}, 0 0 80px {color}, 0 0 90px {color}, 0 0 100px {color}, 0 0 150px {color};
                }}
            }}
            .neon-text {{
                color: #fff;
                text-align: center;
                font-size: {font_size}px;
                font-weight: bold;
                animation: neon-glow 1.5s ease-in-out infinite alternate;
            }}
            </style>
            """

        # HTML for the neon text
        neon_text = f'<div class="neon-text">{text}</div>'

        # Display the CSS and HTML in Streamlit
        st.markdown(neon_style + neon_text, unsafe_allow_html=True)