from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from pywatts.callbacks import LinePlotCallback, CSVCallback
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.modules import FunctionModule, SKLearnWrapper, LinearInterpolater, CalendarExtraction, Sampler, ClockShift, \
    Slicer
from pywatts.summaries import RMSE
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def simple_outlier_filter(x: xr.DataArray):
    """
    Entry is set to None if value is negative or bigger than the 0.99 quantile
    :param x:
    :return:
    """
    filtered = x.values
    filtered[---] = np.NAN  # implement a function to filter extreme values and
                            # assign NAN values (to be interpolated in the next step)

    return numpy_to_xarray(filtered, x, "filtered_outlier")

def debug_fn(x: xr.DataArray):
    """
    Debug function to inspect data during the pipeline execution
    :param x:
    :return:
    """

    return x  # you can place a stop point here to inspect the variable x


if __name__ == "__main__":
    # Load and split data
    data = pd.read_csv("data/getting_started_data.csv", index_col="time", parse_dates=["time"],
                       date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    train_data = ---  # select from start to 2018-6-1
    test_data = ---  #select from 2018-6-1 end

    forecasting_pipeline = Pipeline("results/forecasting")
    # Handle Outlier
    filtered_outlier = FunctionModule(simple_outlier_filter)(x=---)  # select input data
    interpolated_ts = ---  # create linear interpolator step

    # Scale the TS
    scaler = SKLearnWrapper(StandardScaler())
    scaled_ts = scaler(x=interpolated_ts, callbacks=---)  # create line plot callback
    reshaped_ts = FunctionModule(lambda x: numpy_to_xarray(x.values.reshape((-1,)), x, "reshaped"))(x=scaled_ts)

    # Extract the Features
    calendar_features = ---  # extract calendar features
    hist_values = ---  # create -24h lag feature

    # Create samples for forecasting
    calendar_input = ---  # sample calendar feature for a 24h multiple output forecast horizon
    historical_load = ---   # sample lag feature for a 24h multiple output forecast horizon
    target = ---  # sample target for a 24h multiple output forecast horizon

    # Debug step, can be used to inspect the output data of a step, e.g., the sampled calendar feature
    dummy_step = FunctionModule(debug_fn)(x=calendar_input)

    # Cutoff parts with invalid values to the previous sampling
    calendar_input = ---  # slice data to remove zeros from sampling
    historical_load = ---  # slice data to remove zeros from sampling
    target = ---  # slice data to remove zeros from sampling

    # Add the forecasting method
    forecast = SKLearnWrapper(LinearRegression())(x=historical_load, calendar=calendar_input, target=target,
                                                  callbacks=[LinePlotCallback("Predicted Data")])

    rescaled_forecast = scaler(x=forecast, use_inverse_transform=True, computation_mode=ComputationMode.Transform,
                               callbacks=[---])  # create line plot & csv callback
    # Train the pipeline
    ---

    # Add a evaluation metric
    RMSE()(lr_forecast=forecast, y=target)

    # Test the pipeline
    ---
