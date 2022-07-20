from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from pywatts.callbacks import LinePlotCallback
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
    filtered[(np.quantile(x.values, 0.99) < x.values) | (x.values < 0)] = np.NAN

    return numpy_to_xarray(filtered, x)


if __name__ == "__main__":
    # Load and split data
    data = pd.read_csv("data/getting_started_data.csv", index_col="time", parse_dates=["time"],
                       date_parser=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    train_data = data[:pd.Timestamp(year=2018, month=6, day=1)]
    test_data = data[pd.Timestamp(year=2018, month=6, day=1):]

    forecasting_pipeline = Pipeline("results/forecasting")
    # Handle Outlier
    filtered_outlier = FunctionModule(simple_outlier_filter)(x=forecasting_pipeline["load_power_statistics"])
    interpolated_ts = LinearInterpolater()(x=filtered_outlier)

    # Scale the TS
    scaler = SKLearnWrapper(StandardScaler())
    scaled_ts = scaler(x=interpolated_ts, callbacks=[LinePlotCallback("Scaled Data")])
    reshaped_ts = FunctionModule(lambda x: numpy_to_xarray(x.values.reshape((-1,)), x))(x=scaled_ts)

    # Extract the Features
    calendar_features = CalendarExtraction()(x=reshaped_ts)
    hist_values = ClockShift(24)(x=reshaped_ts)

    # Create samples for forecasting
    historical_load = Sampler(sample_size=24)(x=hist_values)
    target = Sampler(24)(x=reshaped_ts)
    calendar_input = Sampler(24)(x=calendar_features)
    # Cutoff parts with invalid values to the previous sampling
    historical_load = Slicer(start=48, end=-48)(x=historical_load)
    target = Slicer(start=48, end=-48)(x=target)
    calendar_input = Slicer(start=48, end=-48)(x=calendar_input)

    # Add the forecasting method
    forecast = SKLearnWrapper(LinearRegression())(x=historical_load, calendar=calendar_input, target=target,
                                                  callbacks=[LinePlotCallback("Predicted Data")])

    rescaled_forecast = scaler(x=forecast, use_inverse_transform=True, computation_mode=ComputationMode.Transform,
                               callbacks=[LinePlotCallback("Rescaled Forecast")])
    # Train the pipeline
    forecasting_pipeline.train(train_data)

    # Add a evaluation metric
    RMSE()(lr_forecast=forecast, y=target)

    # Test the pipeline
    forecasting_pipeline.test(test_data)
