import pandas as pd

from spinesUtils.asserts import ParameterTypeAssert


@ParameterTypeAssert({
    'data1': pd.DataFrame,
    'data2': pd.DataFrame,
    'time_col': str,
    'target_col': str,
    'labels': (None, list, tuple),
    'date_fmt': str
})
def plot_data_period(data1, data2, time_col, target_col, labels=None, date_fmt='%Y-%m-%d'):
    """
    Visualize time-series data, including predictions with upper and lower bounds.

    Parameters
    ----------
    data1 : pd.DataFrame
        First dataset containing time-series data for plotting.
    data2 : pd.DataFrame
        Second dataset containing time-series data for plotting.
    time_col : str
        Column name in the dataframes representing the time information.
    target_col : str
        Column name in the dataframes representing the target variable.
    labels : None or list or tuple, optional, default: None
        Labels for the plot. If None, default labels will be used.
        For predictions with upper and lower bounds, provide labels in the order ('True Data', 'Prediction Data', 'Upper Bound', 'Lower Bound').
        For regular data, provide labels in the order ('Data', 'Test Data').
    date_fmt : str, optional, default: '%Y-%m-%d'
        Date format for the x-axis labels.

    Raises
    ------
    AssertionError
        If the starting time of data1 is after the starting time of data2.

    Notes
    -----
    - The function uses Matplotlib to create a time-series plot with optional upper and lower bounds.
    - The x-axis is formatted with the specified date format.
    - Vertical lines, shaded regions, and grid lines are added for better visualization.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as m_dates

    # Check if the starting time of data1 is before or equal to the starting time of data2
    assert data1[time_col].iloc[0] <= data2[time_col].iloc[0], \
        'The starting time of data1 must be before data2, or equal to the starting time of data2.'

    # Check if labels are provided and in the correct format
    if f'{target_col}_upper' in data2.columns and f'{target_col}_lower' in data2.columns:
        if labels is None:
            labels = ('True Data', 'Prediction Data', 'Upper Bound', 'Lower Bound')
        else:
            assert len(labels) == 4
    else:
        if labels is None:
            labels = ('Data', 'Test Data')
        else:
            assert len(labels) == 2

    # Set x-axis as date format
    date_fmt = m_dates.DateFormatter(date_fmt)
    plt.gca().xaxis.set_major_formatter(date_fmt)

    # Plot curves
    plt.plot_date(data1[time_col], data1[target_col], label=labels[0], color='black', fmt='')
    plt.plot_date(data2[time_col], data2[target_col], label=labels[1], color='blue', fmt='')

    # Plot upper and lower bounds if available
    if f'{target_col}_upper' in data2.columns and  f'{target_col}_lower' in data2.columns:
        plt.plot_date(data2[time_col], data2[f'{target_col}_upper'], label=labels[2], color='orchid', fmt='')
        plt.plot_date(data2[time_col], data2[f'{target_col}_lower'], label=labels[3], color='violet', fmt='')
        plt.fill_between(data2[time_col], data2[f'{target_col}_upper'], data2[f'{target_col}_lower'],
                         color='dimgray', alpha=0.2)

    # Add a vertical line if the starting time of data2 is later than data1
    if data2[time_col].iloc[0] > data1[time_col].iloc[0]:
        plt.axvline(x=data2[time_col].iloc[0], color='black', linestyle='--')

    # Add shaded region indicating the valid period of data2
    valid_data_max = data2[target_col].max()
    valid_period = [data2[time_col].iloc[0], data2[time_col].iloc[-1]]
    plt.axvspan(valid_period[0], valid_period[1], ymin=0, ymax=valid_data_max,
                facecolor='lightyellow', alpha=0.5)

    # Add grid lines
    plt.grid(True)

    # Disable outer box
    plt.box(False)

    # Automatically adjust date label formats and positions to avoid overlap
    plt.gcf().autofmt_xdate()

    # Add legend
    plt.legend()

    # Display the plot
    plt.show()
