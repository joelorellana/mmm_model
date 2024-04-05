"""Demo to show how to use lightweight_mmm in Streamlit.
Author: Joel Orellana
date: 03-apr-2024"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import numpyro
import pickle
from lightweight_mmm import lightweight_mmm
from lightweight_mmm import optimize_media
from lightweight_mmm import plot
from lightweight_mmm import preprocessing
from lightweight_mmm import utils
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import Pipeline   

def budget_allocator(n_weeks_to_predict, budget_to_allocate, mmm, media_scaler, target_scaler, prices):
    """Create a budget allocator function that takes the number of weeks to predict and the budget to allocate as inputs."""
    solution, kpi_without_optim, previous_media_allocation = optimize_media.find_optimal_budgets(
    n_time_periods=n_weeks_to_predict,
    media_mix_model=mmm,
    # extra_features=extra_features_scaled_final[-4:, :],
    budget=budget_to_allocate,
    prices=prices,
    media_scaler=media_scaler,
    target_scaler=target_scaler,
    bounds_lower_pct=0.05,
    bounds_upper_pct=0.95,
    seed=1)
    return solution, kpi_without_optim, previous_media_allocation



# streamlit settings
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>Media Mix Model Analyzer</h1>", unsafe_allow_html=True)
# load a model
model_file = st.file_uploader("Upload a model", type=["pkl"], key="model")
if model_file is not None:
    with st.spinner("Loading model..."):
        try:
            pipeline = pickle.load(model_file)
            mmm = pipeline.named_steps['mmm']
            media_scaler = pipeline.named_steps['media_scaler']
            target_scaler = pipeline.named_steps['target_scaler']
            prices = pipeline.named_steps['prices']
            media_names = pipeline.named_steps['channel_names']
            media_list = [name.replace('impressions_', '') for name in media_names]

            
            # Create columns for the plots only if there is a model
            st.markdown("<h3 style='text-align: center; color: white;'>Model Fit</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.write(plot.plot_model_fit(mmm, target_scaler=target_scaler, digits=2))
            with col2:
                st.write(plot.plot_response_curves(media_mix_model=mmm, target_scaler=target_scaler))
            
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
                            prices
                        )
                        # You can display the results here using st.write() or any other Streamlit functions
                        st.success("Optimal budget allocation calculated.", icon="✅")
                        previous_budget_allocation = prices * previous_media_allocation
                        optimal_budget_allocation = prices * solution.x
                        table_data = pd.DataFrame({
                            'Media': media_list,
                            'Optimal Allocation': optimal_budget_allocation,
                            'Previous Allocation': previous_budget_allocation
                        })
                        # add total to table
                        total_optimal = optimal_budget_allocation.sum()
                        total_previous = previous_budget_allocation.sum()
                        table_data.loc[len(table_data)] = ['Total', total_optimal, total_previous]
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
                                                channel_names = media_list,
                                                ))

                    except Exception as e:
                        st.error(f"Failed to run budget allocator: {e}") 
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()
else:
    st.error("Please upload a model file to proceed.")


