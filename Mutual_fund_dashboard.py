from mftool import Mftool
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

mf = Mftool()

st.title("Mutual Fund Financial Dashboard")

option = st.sidebar.selectbox(
    "Choose an action",
    [
        "View Available Schemes",
        "Scheme Details",
        "Historical NAV",
        "Compare NAVs",
        "Average AUM",
        "Performance Heatmap",
        "Risk and Volatility Analysis"
    ]
)

# Safe wrapper to get scheme codes
try:
    scheme_codes_raw = mf.get_scheme_codes()
    scheme_names = {v: k for k, v in scheme_codes_raw.items()}
except requests.exceptions.RequestException as e:
    st.error("Error fetching scheme codes.")
    st.stop()
except ValueError as e:
    st.error("Invalid response received when fetching scheme codes.")
    st.stop()

if option == "View Available Schemes":
    st.header("View Available Schemes")
    amc = st.sidebar.text_input("Enter AMC Name", "ICICI")

    try:
        schemes = mf.get_available_schemes(amc)
        if schemes:
            st.write(pd.DataFrame(schemes.items(), columns=["Scheme Code", "Scheme Name"]))
        else:
            st.warning("No schemes found.")
    except requests.exceptions.RequestException:
        st.error("Failed to fetch schemes. Please check your connection or AMC name.")
    except ValueError:
        st.error("Invalid response from server when fetching available schemes.")

elif option == "Scheme Details":
    st.header("Scheme Details")

    try:
        selected_scheme = st.sidebar.selectbox("Select a Scheme", list(scheme_names.keys()))
        scheme_code = scheme_names[selected_scheme]
        details = pd.DataFrame(mf.get_scheme_details(scheme_code)).iloc[0]
        st.write(details)
    except Exception as e:
        st.error(f"Failed to fetch scheme details: {e}")

elif option == "Historical NAV":
    st.header("Historical NAV")
    scheme_code = scheme_names[st.sidebar.selectbox("Select a Scheme", scheme_names.keys())]
    nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
    st.write(nav_data)

    
# COMPARE NAV
elif option == "Compare NAVs":
    st.header("Compare NAVs")
    selected_schemes = st.sidebar.multiselect("Select Schemes to Compare", options=list(scheme_names.keys()))

    aggregation = st.sidebar.radio("Select Data Frequency", ["Daily", "Weekly", "Monthly"])

    if selected_schemes:
        nav_frames = []

        for scheme in selected_schemes:
            code = scheme_names[scheme]
            data = mf.get_scheme_historical_nav(code, as_Dataframe=True)
            if data.empty:
                st.warning(f"No data found for {scheme}. Skipping.")
                continue

            data = data.reset_index().rename(columns={"index": "date"})
            data["date"] = pd.to_datetime(data["date"], dayfirst=True)
            data = data.sort_values("date")

            # Convert NAV to numeric and clean
            data["nav"] = pd.to_numeric(data["nav"], errors="coerce").replace(0, pd.NA)

            # Apply aggregation
            if aggregation == "Weekly":
                data = data.set_index("date").resample("W-FRI").last().reset_index()
            elif aggregation == "Monthly":
                data = data.set_index("date").resample("M").last().reset_index()

            data = data[["date", "nav"]].rename(columns={"nav": scheme})
            nav_frames.append(data)

        if nav_frames:
            comparison_df = nav_frames[0]
            for df in nav_frames[1:]:
                comparison_df = pd.merge(comparison_df, df, on="date", how="outer")

            comparison_df = comparison_df.sort_values("date")

            # Create Plotly line chart
            fig = px.line(
                comparison_df,
                x="date",
                y=selected_schemes,
                title=f"Comparison of NAVs - {aggregation} Data",
                labels={"value": "NAV", "date": "Date", "variable": "Scheme"},
                markers=True
            )

            # Add Y-axis padding
            y_min = comparison_df[selected_schemes].min().min()
            y_max = comparison_df[selected_schemes].max().max()
            if pd.notna(y_min) and pd.notna(y_max):
                fig.update_yaxes(
                    range=[y_min - 0.5, y_max + 0.5],
                    showspikes=True,
                    spikemode='across',
                    spikethickness=1,
                    spikecolor="gray",
                )

            # Stylish X-axis range slider + Y-axis slider
            fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(
                        visible=True,
                        thickness=0.005,  # ultra-thin
                        bgcolor="lightgray"
                    ),
                    showspikes=True,
                    spikemode='across',
                    spikecolor="gray",
                    spikethickness=1,
                    type="date"
                ),
                yaxis=dict(
                    fixedrange=False,  # allow zoom and drag on y-axis
                    showspikes=True,
                    spikemode='across',
                    spikecolor="gray",
                    spikethickness=1,
                    tickformat=".2f",
                ),
                dragmode='zoom',  # allows box zoom and pan
                hovermode="x unified",
                margin=dict(l=30, r=30, t=40, b=30),
                template="plotly_white",
                height=600,
                )


            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No valid data to display.")
    else:
        st.info("Select at least one scheme.")





elif option == "Average AUM":
    st.header("Average AUM")
    aum_data = mf.get_average_aum("July September 2024", False)

    if aum_data:
        aum_df = pd.DataFrame(aum_data)
        aum_df["Total AUM"] = aum_df[["AAUM Overseas", "AAUM Domestic"]].astype(float).sum(axis=1)
        st.write(aum_df[["Fund Name", "Total AUM"]])
    else:
        st.write("No AUM data available.")

elif option == "Performance Heatmap":
    st.header("Performance Heatmap")
    scheme_code = scheme_names[st.sidebar.selectbox("Select a Scheme", scheme_names.keys())]
    nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)

    if not nav_data.empty:
        nav_data = nav_data.reset_index().rename(columns={"index": "date"})
        nav_data["date"] = pd.to_datetime(nav_data["date"], dayfirst=True)
        nav_data["nav"] = pd.to_numeric(nav_data["nav"], errors="coerce")
        nav_data = nav_data.dropna(subset=["nav"])

        # Sort by date ascending
        nav_data = nav_data.sort_values("date")

        # Calculate daily returns
        nav_data["returns"] = nav_data["nav"].pct_change()

        # Extract year and month
        nav_data["year"] = nav_data["date"].dt.year
        nav_data["month"] = nav_data["date"].dt.strftime('%b')

        # Monthly returns
        monthly_nav = nav_data.groupby(["year", "month"]).agg({"nav": ["first", "last"]})
        monthly_nav.columns = ["nav_first", "nav_last"]
        monthly_nav = monthly_nav.reset_index()
        monthly_nav["monthly_return"] = (monthly_nav["nav_last"] / monthly_nav["nav_first"]) - 1

        # Pivot for heatmap
        heatmap_data = monthly_nav.pivot(index="year", columns="month", values="monthly_return")

        # Month order
        months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        heatmap_data = heatmap_data.reindex(columns=months_order)

        # Heatmap
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale=["#8B0000", "#FF4500", "#FFD700", "#ADFF2F", "#006400"],
            labels=dict(x="Month", y="Year", color="Monthly Return"),
            x=months_order,
            y=heatmap_data.index,
            title=f"Monthly Return Heatmap for {scheme_code}",
            text_auto=".2%"
        )

        # Dark theme & white text labels
        fig.update_traces(textfont=dict(color="white"))
        fig.update_layout(
            yaxis_autorange="reversed",
            plot_bgcolor="#222222",
            paper_bgcolor="#222222",
            font=dict(color="white")
        )

        st.plotly_chart(fig)
    else:
        st.write("No data available for heatmap.")
elif option == "Performance Heatmap":
    st.header("Performance Heatmap")
    scheme_code = scheme_names[st.sidebar.selectbox("Select a Scheme", scheme_names.keys())]
    nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)

    if not nav_data.empty:
        nav_data = nav_data.reset_index().rename(columns={"index": "date"})
        nav_data["date"] = pd.to_datetime(nav_data["date"], dayfirst=True)
        nav_data["nav"] = pd.to_numeric(nav_data["nav"], errors="coerce")
        nav_data = nav_data.dropna(subset=["nav"])

        # Sort by date ascending
        nav_data = nav_data.sort_values("date")

        # Calculate daily returns
        nav_data["returns"] = nav_data["nav"].pct_change()

        # Extract year and month
        nav_data["year"] = nav_data["date"].dt.year
        nav_data["month"] = nav_data["date"].dt.strftime('%b')

        # Monthly returns
        monthly_nav = nav_data.groupby(["year", "month"]).agg({"nav": ["first", "last"]})
        monthly_nav.columns = ["nav_first", "nav_last"]
        monthly_nav = monthly_nav.reset_index()
        monthly_nav["monthly_return"] = (monthly_nav["nav_last"] / monthly_nav["nav_first"]) - 1

        # Pivot for heatmap
        heatmap_data = monthly_nav.pivot(index="year", columns="month", values="monthly_return")

        # Month order
        months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        heatmap_data = heatmap_data.reindex(columns=months_order)

        # Heatmap
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale=["#8B0000", "#FF4500", "#FFD700", "#ADFF2F", "#006400"],
            labels=dict(x="Month", y="Year", color="Monthly Return"),
            x=months_order,
            y=heatmap_data.index,
            title=f"Monthly Return Heatmap for {scheme_code}",
            text_auto=".2%"
        )

        # Dark theme & white text labels
        fig.update_traces(textfont=dict(color="white"))
        fig.update_layout(
            yaxis_autorange="reversed",
            plot_bgcolor="#222222",
            paper_bgcolor="#222222",
            font=dict(color="white")
        )

        st.plotly_chart(fig)
    else:
        st.write("No data available for heatmap.")
elif option == "Performance Heatmap":
    st.header("Performance Heatmap")
    scheme_code = scheme_names[st.sidebar.selectbox("Select a Scheme", scheme_names.keys())]
    nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)

    if not nav_data.empty:
        nav_data = nav_data.reset_index().rename(columns={"index": "date"})
        nav_data["date"] = pd.to_datetime(nav_data["date"], dayfirst=True)
        nav_data["nav"] = pd.to_numeric(nav_data["nav"], errors="coerce")
        nav_data = nav_data.dropna(subset=["nav"])

        # Sort by date ascending
        nav_data = nav_data.sort_values("date")

        # Calculate daily returns
        nav_data["returns"] = nav_data["nav"].pct_change()

        # Extract year and month
        nav_data["year"] = nav_data["date"].dt.year
        nav_data["month"] = nav_data["date"].dt.strftime('%b')

        # Monthly returns
        monthly_nav = nav_data.groupby(["year", "month"]).agg({"nav": ["first", "last"]})
        monthly_nav.columns = ["nav_first", "nav_last"]
        monthly_nav = monthly_nav.reset_index()
        monthly_nav["monthly_return"] = (monthly_nav["nav_last"] / monthly_nav["nav_first"]) - 1

        # Pivot for heatmap
        heatmap_data = monthly_nav.pivot(index="year", columns="month", values="monthly_return")

        # Month order
        months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        heatmap_data = heatmap_data.reindex(columns=months_order)

        # Heatmap
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale=["#8B0000", "#FF4500", "#FFD700", "#ADFF2F", "#006400"],
            labels=dict(x="Month", y="Year", color="Monthly Return"),
            x=months_order,
            y=heatmap_data.index,
            title=f"Monthly Return Heatmap for {scheme_code}",
            text_auto=".2%"
        )

        # Dark theme & white text labels
        fig.update_traces(textfont=dict(color="white"))
        fig.update_layout(
            yaxis_autorange="reversed",
            plot_bgcolor="#222222",
            paper_bgcolor="#222222",
            font=dict(color="white")
        )

        st.plotly_chart(fig)
    else:
        st.write("No data available for heatmap.")
elif option == "Performance Heatmap":
    st.header("Performance Heatmap")
    scheme_code = scheme_names[st.sidebar.selectbox("Select a Scheme", scheme_names.keys())]
    nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)

    if not nav_data.empty:
        nav_data = nav_data.reset_index().rename(columns={"index": "date"})
        nav_data["date"] = pd.to_datetime(nav_data["date"], dayfirst=True)
        nav_data["nav"] = pd.to_numeric(nav_data["nav"], errors="coerce")
        nav_data = nav_data.dropna(subset=["nav"])

        # Sort by date ascending
        nav_data = nav_data.sort_values("date")

        # Calculate daily returns
        nav_data["returns"] = nav_data["nav"].pct_change()

        # Extract year and month
        nav_data["year"] = nav_data["date"].dt.year
        nav_data["month"] = nav_data["date"].dt.strftime('%b')

        # Monthly returns
        monthly_nav = nav_data.groupby(["year", "month"]).agg({"nav": ["first", "last"]})
        monthly_nav.columns = ["nav_first", "nav_last"]
        monthly_nav = monthly_nav.reset_index()
        monthly_nav["monthly_return"] = (monthly_nav["nav_last"] / monthly_nav["nav_first"]) - 1

        # Pivot for heatmap
        heatmap_data = monthly_nav.pivot(index="year", columns="month", values="monthly_return")

        # Month order
        months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        heatmap_data = heatmap_data.reindex(columns=months_order)

        # Heatmap
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale=["#8B0000", "#FF4500", "#FFD700", "#ADFF2F", "#006400"],
            labels=dict(x="Month", y="Year", color="Monthly Return"),
            x=months_order,
            y=heatmap_data.index,
            title=f"Monthly Return Heatmap for {scheme_code}",
            text_auto=".2%"
        )

        # Dark theme & white text labels
        fig.update_traces(textfont=dict(color="white"))
        fig.update_layout(
            yaxis_autorange="reversed",
            plot_bgcolor="#222222",
            paper_bgcolor="#222222",
            font=dict(color="white")
        )

        st.plotly_chart(fig)
    else:
        st.write("No data available for heatmap.")
elif option == "Performance Heatmap":
    st.header("Performance Heatmap")
    scheme_code = scheme_names[st.sidebar.selectbox("Select a Scheme", scheme_names.keys())]
    nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)

    if not nav_data.empty:
        nav_data = nav_data.reset_index().rename(columns={"index": "date"})
        nav_data["date"] = pd.to_datetime(nav_data["date"], dayfirst=True)
        nav_data["nav"] = pd.to_numeric(nav_data["nav"], errors="coerce")
        nav_data = nav_data.dropna(subset=["nav"])

        # Sort by date ascending
        nav_data = nav_data.sort_values("date")

        # Calculate daily returns
        nav_data["returns"] = nav_data["nav"].pct_change()

        # Extract year and month
        nav_data["year"] = nav_data["date"].dt.year
        nav_data["month"] = nav_data["date"].dt.strftime('%b')

        # Monthly returns
        monthly_nav = nav_data.groupby(["year", "month"]).agg({"nav": ["first", "last"]})
        monthly_nav.columns = ["nav_first", "nav_last"]
        monthly_nav = monthly_nav.reset_index()
        monthly_nav["monthly_return"] = (monthly_nav["nav_last"] / monthly_nav["nav_first"]) - 1

        # Pivot for heatmap
        heatmap_data = monthly_nav.pivot(index="year", columns="month", values="monthly_return")

        # Month order
        months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        heatmap_data = heatmap_data.reindex(columns=months_order)

        # Heatmap
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale=["#8B0000", "#FF4500", "#FFD700", "#ADFF2F", "#006400"],
            labels=dict(x="Month", y="Year", color="Monthly Return"),
            x=months_order,
            y=heatmap_data.index,
            title=f"Monthly Return Heatmap for {scheme_code}",
            text_auto=".2%"
        )

        # Dark theme & white text labels
        fig.update_traces(textfont=dict(color="white"))
        fig.update_layout(
            yaxis_autorange="reversed",
            plot_bgcolor="#222222",
            paper_bgcolor="#222222",
            font=dict(color="white")
        )

        st.plotly_chart(fig)
    else:
        st.write("No data available for heatmap.")
elif option == "Performance Heatmap":
    st.header("Performance Heatmap")
    scheme_code = scheme_names[st.sidebar.selectbox("Select a Scheme", scheme_names.keys())]
    nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)

    if not nav_data.empty:
        nav_data = nav_data.reset_index().rename(columns={"index": "date"})
        nav_data["date"] = pd.to_datetime(nav_data["date"], dayfirst=True)
        nav_data["nav"] = pd.to_numeric(nav_data["nav"], errors="coerce")
        nav_data = nav_data.dropna(subset=["nav"])

        # Sort by date ascending
        nav_data = nav_data.sort_values("date")

        # Calculate daily returns
        nav_data["returns"] = nav_data["nav"].pct_change()

        # Extract year and month
        nav_data["year"] = nav_data["date"].dt.year
        nav_data["month"] = nav_data["date"].dt.strftime('%b')

        # Monthly returns
        monthly_nav = nav_data.groupby(["year", "month"]).agg({"nav": ["first", "last"]})
        monthly_nav.columns = ["nav_first", "nav_last"]
        monthly_nav = monthly_nav.reset_index()
        monthly_nav["monthly_return"] = (monthly_nav["nav_last"] / monthly_nav["nav_first"]) - 1

        # Pivot for heatmap
        heatmap_data = monthly_nav.pivot(index="year", columns="month", values="monthly_return")

        # Month order
        months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        heatmap_data = heatmap_data.reindex(columns=months_order)

        # Heatmap
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale=["#8B0000", "#FF4500", "#FFD700", "#ADFF2F", "#006400"],
            labels=dict(x="Month", y="Year", color="Monthly Return"),
            x=months_order,
            y=heatmap_data.index,
            title=f"Monthly Return Heatmap for {scheme_code}",
            text_auto=".2%"
        )

        # Dark theme & white text labels
        fig.update_traces(textfont=dict(color="white"))
        fig.update_layout(
            yaxis_autorange="reversed",
            plot_bgcolor="#222222",
            paper_bgcolor="#222222",
            font=dict(color="white")
        )

        st.plotly_chart(fig)
    else:
        st.write("No data available for heatmap.")

elif option == "Performance Heatmap":
    st.header("Performance Heatmap")
    scheme_code = scheme_names[st.sidebar.selectbox("Select a Scheme", scheme_names.keys())]
    nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)

    if not nav_data.empty:
        nav_data = nav_data.reset_index().rename(columns={"index": "date"})
        nav_data["date"] = pd.to_datetime(nav_data["date"], dayfirst=True)
        nav_data["nav"] = pd.to_numeric(nav_data["nav"], errors="coerce")
        nav_data = nav_data.dropna(subset=["nav"])

        # Sort by date ascending for returns calculation
        nav_data = nav_data.sort_values("date")

        # Calculate daily returns
        nav_data["returns"] = nav_data["nav"].pct_change()

        # Extract year and month abbreviation
        nav_data["year"] = nav_data["date"].dt.year
        nav_data["month"] = nav_data["date"].dt.strftime('%b')  # Jan, Feb, etc.

        # Calculate monthly returns
        monthly_nav = nav_data.groupby(["year", "month"]).agg({"nav": ["first", "last"]})
        monthly_nav.columns = ["nav_first", "nav_last"]
        monthly_nav = monthly_nav.reset_index()
        monthly_nav["monthly_return"] = (monthly_nav["nav_last"] / monthly_nav["nav_first"]) - 1

        # Pivot for heatmap
        heatmap_data = monthly_nav.pivot(index="year", columns="month", values="monthly_return")

        # Reorder columns to calendar months
        months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        heatmap_data = heatmap_data.reindex(columns=months_order)

        # Plot heatmap
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale=[
                [0.0, "#8B0000"],  # Dark Red
                [0.25, "#FF4500"], # Orange-Red
                [0.5, "#FFD700"],  # Yellow (Neutral)
                [0.75, "#ADFF2F"], # Light Green
                [1.0, "#006400"]   # Dark Green
            ],

            labels=dict(x="Month", y="Year", color="Monthly Return"),
            x=months_order,
            y=heatmap_data.index,
            title=f"Monthly Return Heatmap for {scheme_code}",
            text_auto=".2%"
        )

        # Make all text white (inside cells & axis labels)
        fig.update_traces(textfont=dict(color="white"))
        fig.update_layout(
            yaxis_autorange="reversed",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color="white"),  # Changes axis labels, title, legend to white
            xaxis=dict(tickfont=dict(color="black")),
            yaxis=dict(tickfont=dict(color="black"))
        )

        st.plotly_chart(fig)
    else:
        st.write("No data available for heatmap.")

elif option == "Risk and Volatility Analysis":
    st.header("Risk and Volatility Analysis")
    scheme_name = st.sidebar.selectbox("Select a Scheme", scheme_names.keys())
    scheme_code = scheme_names[scheme_name]
    nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)

    if not nav_data.empty:
        # Reset and clean data
        nav_data = nav_data.reset_index().rename(columns={"index": "date"})
        nav_data["date"] = pd.to_datetime(nav_data["date"], dayfirst=True)
        nav_data["nav"] = pd.to_numeric(nav_data["nav"], errors="coerce")
        nav_data = nav_data.dropna(subset=["nav"])

        # Sort from newest to oldest
        nav_data = nav_data.sort_values("date", ascending=False)

        # Calculate correct daily returns
        nav_data["returns"] = nav_data["nav"].pct_change()
        nav_data = nav_data.dropna(subset=["returns"])

        # Use latest NAV for simulation
        last_nav = nav_data["nav"].iloc[0]

        # Calculate metrics
        annualized_volatility = nav_data["returns"].std() * np.sqrt(252)
        annualized_return = (1 + nav_data["returns"].mean()) ** 252 - 1
        risk_free_rate = 0.06
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

        st.write(f"### Metrics for {scheme_name}")
        st.metric("Annualized Volatility", f"{annualized_volatility:.2%}")
        st.metric("Annualized Return", f"{annualized_return:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

        # Plot Risk-Return Distribution
        fig = px.scatter(
            nav_data,
            x="date",
            y="returns",
            title=f"Risk-Return Scatter for {scheme_name}",
            labels={"returns": "Daily Returns", "date": "Date"}
        )
        st.plotly_chart(fig)

        # Monte Carlo Simulation for Future NAV Projection
        st.write("### Monte Carlo Simulation for Future NAV Projection")

        # Simulation Parameters
        num_simulations = st.slider("Number of Simulations", min_value=100, max_value=5000, value=1000)
        num_days = st.slider("Projection Period (Days)", min_value=30, max_value=365, value=252)

        # Monte Carlo simulation
        daily_volatility = nav_data["returns"].std()
        daily_mean_return = nav_data["returns"].mean()
        simulation_results = []

        for _ in range(num_simulations):
            prices = [last_nav]
            for _ in range(num_days):
                simulated_return = np.random.normal(daily_mean_return, daily_volatility)
                next_price = prices[-1] * (1 + simulated_return)
                prices.append(next_price)
            simulation_results.append(prices)

        # Create DataFrame for Visualization
        simulation_df = pd.DataFrame(simulation_results).T
        simulation_df.index.name = "Day"
        simulation_df.columns = [f"Simulation {i+1}" for i in range(num_simulations)]

        # Plot Simulations
        fig_simulation = px.line(
            simulation_df,
            title=f"Monte Carlo Simulation for {scheme_name} NAV Projection",
            labels={"value": "Projected NAV", "Day": "Day"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_simulation)

        # Show Summary Statistics
        final_prices = simulation_df.iloc[-1]
        st.write(f"### Simulation Summary for {scheme_name}")
        st.metric("Expected Final NAV", f"{final_prices.mean():.2f}")
        st.metric("Minimum Final NAV", f"{final_prices.min():.2f}")
        st.metric("Maximum Final NAV", f"{final_prices.max():.2f}")

    else:
        st.write("No historical NAV data available.")
