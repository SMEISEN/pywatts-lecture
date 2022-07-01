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
    # !! Task 1 !!
    # implement a function to filter extreme values and
    # assign NAN values (to be interpolated in the next step)
    filtered[---] = np.NAN

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

    # !! Task 2 !!
    # Create a train data set with data select from the start of the full data set until to 2018-6-1
    train_data = ---

    # !! Task 3 !!
    # Create a test data set with data from 2018-6-1 to the end of the full data set
    test_data = ---

    forecasting_pipeline = Pipeline("results/forecasting")
    # Handle Outlier

    # !! Task 4 !!
    # select input data as the column "load_power_statistics"
    filtered_outlier = FunctionModule(simple_outlier_filter)(x=---)

    # !! Task 5 !!
    # Include a linear interpolator step (check the pyWATTS documentation)
    interpolated_ts = ---

    # Scale the TS
    scaler = SKLearnWrapper(StandardScaler())

    # !! Task 6 !!
    # Include a line plot callback
    scaled_ts = scaler(x=interpolated_ts, callbacks=---)
    reshaped_ts = FunctionModule(lambda x: numpy_to_xarray(x.values.reshape((-1,)), x, "reshaped"))(x=scaled_ts)

    # Extract the Features

    # !! Task 7 !!
    # extract calendar features (check the pyWATTS documentation)
    calendar_features = ---

    # !! Task 8 !!
    # Shift the original time series 24h to create a -24h lag feature
    hist_values = ---

    # Create samples for forecasting

    # !! Task 9 !!
    # Create a sample of the calendar features for a 24h multiple output forecast horizon
    calendar_input = ---

    # !! Task 10 !!
    # Create a sample of the historical load lag features for a 24h multiple output forecast horizon
    historical_load = ---

    # !! Task 11 !!
    # Create a sample of the target data for a 24h multiple output forecast horizon
    target = ---

    # Debug step, can be used to inspect the output data of a step, e.g., the sampled calendar feature
    dummy_step = FunctionModule(debug_fn)(x=calendar_input)

    # Cutoff parts with invalid values to the previous sampling

    # !! Task 12 !!
    # slice the calendar_input data to remove zeros from sampling
    calendar_input = ---

    # !! Task 13 !!
    # slice the historical data to remove zeros from sampling
    historical_load = ---

    # !! Task 14 !!
    # Slice the target data to remove zeros from sampling
    target = ---

    # Add the forecasting method
    forecast = SKLearnWrapper(LinearRegression())(x=historical_load, calendar=calendar_input, target=target,
                                                  callbacks=[LinePlotCallback("Predicted Data")])

    # !! Task 15 !!
    # Include a line plot & a csv callback
    rescaled_forecast = scaler(x=forecast, use_inverse_transform=True, computation_mode=ComputationMode.Transform,
                               callbacks=[---])


    # !! Task 16 !!
    # Write the command to train the pipeline
    ---

    # Add a evaluation metric
    RMSE()(lr_forecast=forecast, y=target)


    # !! Task 17 !!
    # Write the command to test the pipeline
    ---
