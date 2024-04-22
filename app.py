"""Demo to show how to use lightweight_mmm in Streamlit.
Author: Joel Orellana
last update: 22-apr-2024"""

import io
import numpyro
import holidays
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import tempfile
from datetime import timedelta
from lightweight_mmm import lightweight_mmm
from lightweight_mmm import optimize_media
from lightweight_mmm import plot
from lightweight_mmm import preprocessing
from lightweight_mmm import utils
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from google.cloud import storage
from st_files_connection import FilesConnection



# streamlit settings
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>Media Mix Model Analyzer</h1>", unsafe_allow_html=True)
BUCKET_NAME = 'a0-mmm-models'
CLIENT = storage.Client() # Initialize a client for Google Cloud Storage
st.info("CURRENTLY IN CONSTRUCTION.")




def add_holiday_columns_to_array(end_date,time_period):
    df = pd.DataFrame({'week': [pd.to_datetime(end_date) + pd.DateOffset(weeks=x) for x in range(1, time_period + 1)]})
    # Ensure that 'week' column is in datetime format
    df['week'] = pd.to_datetime(df['week'])
    us_holidays = holidays.US()
    # Extract the years from the 'week' column to determine which years we need holiday data for
    unique_years = df['week'].dt.year.unique()

    # Create a dictionary to store holidays
    holiday_dict = {}
    for date, name in sorted(holidays.US(years=unique_years).items()):
        # Standardize the holiday name
        standardized_name = "hldy_" + name.split(' (')[0].replace("'s", "").replace(".", "").replace(" ", "_").lower()
        holiday_dict.setdefault(standardized_name, set()).add(date)

    # Initialize columns for each holiday with 0
    for holiday in holiday_dict:
        df[holiday] = 0

    # Apply the holiday check to each row and update the DataFrame accordingly
    for index, row in df.iterrows():
        week_start_date = row['week'].date()
        week_end = week_start_date + timedelta(days=6)

        for holiday, dates in holiday_dict.items():
            df.at[index, holiday] = int(any(week_start_date <= holiday_date <= week_end for holiday_date in dates))

    # Remove date from extra features
    df.drop('week', axis=1, inplace=True)

    # transform to numpy
    df = df.to_numpy()
    # scaling
    df = extra_scaler.transform(df)
    return df


def budget_allocator(n_weeks_to_predict, budget_to_allocate, mmm, media_scaler, target_scaler, prices, end_date):
    """Create a budget allocator function that takes the number of weeks to predict and the budget to allocate as inputs."""
    solution, kpi_without_optim, previous_media_allocation = optimize_media.find_optimal_budgets(
    n_time_periods=n_weeks_to_predict,
    media_mix_model=mmm,
    extra_features=add_holiday_columns_to_array(end_date, n_weeks_to_predict),
    budget=budget_to_allocate,
    prices=prices,
    media_scaler=media_scaler,
    target_scaler=target_scaler,
    bounds_lower_pct=0.05,
    bounds_upper_pct=0.95,
    seed=1)
    return solution, kpi_without_optim, previous_media_allocation

def select_model():
    bucket = CLIENT.get_bucket(BUCKET_NAME) # Get the bucket object
    blobs = bucket.list_blobs() # List all objects in the bucket
    pkl_files_list = [blob.name for blob in blobs if blob.name.endswith('.pkl')]
    print(pkl_files_list)
    return pkl_files_list

@st.cache_data
def read_model(selected_model):
    # get a copy in current directory of the mmm model
    blob_name = f"{selected_model}"
    bucket = CLIENT.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    file_path = f"{temp_dir}/{select_model}"
    
    # Download the model file to the temporary directory
    blob.download_to_filename(file_path)
    return file_path



@st.cache_data(ttl=30)
def load_model(model_path):
    model_file = open(model_path, 'rb')
    pipeline = pickle.load(model_file)
    name_model = pipeline.named_steps['name_model']
    start_date = pipeline.named_steps["start_date"]
    end_date = pipeline.named_steps["end_date"]
    media_scaler = pipeline.named_steps['media_scaler']
    target_scaler = pipeline.named_steps['target_scaler']
    media_names = pipeline.named_steps['channel_names']
    cost_scaler = pipeline.named_steps['cost_scaler']
    extra_scaler = pipeline.named_steps['extra_scaler']
    prices = pipeline.named_steps['prices']
    mmm = pipeline.named_steps['mmm']
    media_contribution, roi_hat = mmm.get_posterior_metrics(target_scaler=target_scaler, cost_scaler=cost_scaler)
    plot1 = plot.plot_model_fit(mmm, target_scaler=target_scaler, digits=2)
    plot2 = plot.plot_response_curves(media_mix_model=mmm, target_scaler=target_scaler)
    plot3 = plot.plot_media_baseline_contribution_area_plot(media_mix_model=mmm,
                                                target_scaler=target_scaler,
                                                fig_size=(30,10),
                                                channel_names = media_names
                                                )
    plot4 = plot.plot_bars_media_metrics(metric=roi_hat, metric_name=" ROI hat", channel_names=media_names)
    plot5 = plot.plot_bars_media_metrics(metric=media_contribution, metric_name="Media Contribution Percentage", channel_names=media_names)

    return name_model, start_date, end_date, media_scaler, target_scaler, media_names, cost_scaler, extra_scaler, prices, mmm, plot1, plot2, plot3, plot4, plot5






model_list = select_model()
selected_model = st.selectbox("Select a model from the list:", model_list, placeholder="Select a model", index=None) # default None
if selected_model:
    with st.spinner("Loading model..."):
        temp_file_model_path = read_model(selected_model)
        name_model, start_date, end_date, media_scaler, target_scaler, media_names, cost_scaler, extra_scaler, prices, mmm, plot1, plot2, plot3, plot4, plot5 = load_model(temp_file_model_path)
        st.success(f"Model {name_model} loaded successfully!")
        # adding model info 
        st.info(f"Model was trained using data from {start_date} to {end_date} for {', '.join(media_names)}.")
        # Create columns for the plots only if there is a model
        st.markdown("<h3 style='text-align: center; color: white;'>Model Report</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write(plot1)
        with col2:
            st.write(plot2)
        st.write(plot3)
        col3, col4 = st.columns(2)
        with col3:
            st.write(plot4)
        with col4:
            st.write(plot5)
            
    # new section Budget Allocator Predictor
    st.markdown("<h3 style='text-align: center; color: white;'>Budget Estimator</h3>", unsafe_allow_html=True)

    # Layout: Number of weeks, Budget, and Button in one row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        n_weeks_to_predict = st.number_input("Number of weeks to predict:", min_value=1, max_value=12, step=1, key='weeks_input')
    with col3:
        budget_to_allocate = st.number_input("Budget to allocate:", step=1000, min_value=1000, key='budget_input')
    with col5:
        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button('Run Budget Allocator')

    # Button to trigger the budget allocator
    if run_button:
        with st.spinner("Calculating optimal budget allocation..."):
            try:
                solution, kpi_without_optim, previous_media_allocation = budget_allocator(
                    n_weeks_to_predict, 
                    budget_to_allocate, 
                    mmm, 
                    media_scaler, 
                    target_scaler, 
                    prices,
                    end_date=end_date,
                )
                previous_budget_allocation = round(prices * previous_media_allocation, 2)
                optimal_budget_allocation = round(prices * solution.x, 2)
                table_data = pd.DataFrame({
                    'Media': media_names,
                    'Optimal Allocation': optimal_budget_allocation,
                    'Previous Allocation': previous_budget_allocation
                })
                # Convert 'Optimal Allocation' to a float64 type
                table_data['Optimal Allocation'] = table_data['Optimal Allocation'].astype('float64')
                # Format the 'Optimal Allocation' column to two decimal places
                table_data['Optimal Allocation'] = table_data['Optimal Allocation'].round(2)
                # Convert 'Previous Allocation' to a float64 type
                table_data['Previous Allocation'] = table_data['Previous Allocation'].astype('float64')
                # Format the 'Previous Allocation' column to two decimal places
                table_data['Previous Allocation'] = table_data['Previous Allocation'].round(2)
                # add total to table
                total_optimal = optimal_budget_allocation.sum()
                total_previous = previous_budget_allocation.sum()
                table_data.loc[len(table_data)] = ['Total', total_optimal, total_previous]
                # Display the table only if total_optimal and total_previous are approximately equal
                if abs(total_optimal - total_previous) < 10:
                    st.success("Optimal budget allocation calculated.", icon="✅")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(table_data)
                    with col2:
                        st.write(plot.plot_pre_post_budget_allocation_comparison(media_mix_model=mmm,
                                            kpi_with_optim=solution['fun'],
                                            kpi_without_optim=kpi_without_optim,
                                            optimal_buget_allocation=optimal_budget_allocation,
                                            previous_budget_allocation=previous_budget_allocation,
                                            figure_size=(10,8),
                                            channel_names = media_names,
                                            ))
                else: # raise an streamlit error indicating that the budget is not realistic with the historical data
                    st.error("The budget is not realistic with the historical data. Please change the budget or the number of weeks to predict.")

            except Exception as e:
                st.error(f"Failed to run budget allocator: {e}") 

else:
    st.error("Please select a trained model to proceed.")
