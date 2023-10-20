import pandas as pd

from spinesUtils.asserts import ParameterTypeAssert, ParameterValuesAssert


@ParameterTypeAssert({
    'data1': pd.DataFrame,
    'data2': pd.DataFrame,
    'time_col': str,
    'target_col': str,
    'labels': (None, list, tuple),
    'date_fmt': str
})
def plot_data_period(data1, data2, time_col, target_col, labels=None, date_fmt='%Y-%m-%d'):
    """可视化日期数据"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as m_dates

    assert data1[time_col].iloc[0] <= data2[time_col].iloc[0], \
        'The starting time of data1 must be before data2, or equal to the starting time of data2.'

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

    # 设置x轴为日期格式
    date_fmt = m_dates.DateFormatter(date_fmt)
    plt.gca().xaxis.set_major_formatter(date_fmt)

    # 绘制曲线
    plt.plot_date(data1[time_col], data1[target_col], label=labels[0],
                  color='black', linestyle='-', marker='')
    plt.plot_date(data2[time_col], data2[target_col],
                  label=labels[1], color='blue', linestyle='-', marker='')

    if f'{target_col}_upper' in data2.columns and  f'{target_col}_lower' in data2.columns:
        plt.plot_date(data2[time_col], data2[f'{target_col}_upper'],
                      label=labels[2], color='orchid', linestyle='-', marker='')
        plt.plot_date(data2[time_col], data2[f'{target_col}_lower'],
                      label=labels[3], color='violet', linestyle='-', marker='')
        plt.fill_between(data2[time_col], data2[f'{target_col}_upper'], data2[f'{target_col}_lower'],
                         color='dimgray', alpha=0.2)

    if data2[time_col].iloc[0] > data1[time_col].iloc[0]:
        # 添加竖线
        plt.axvline(x=data2[time_col].iloc[0], color='black', linestyle='--')

    valid_data_max = data2[target_col].max()
    valid_period = [data2[time_col].iloc[0], data2[time_col].iloc[-1]]

    plt.axvspan(valid_period[0], valid_period[1], ymin=0, ymax=valid_data_max,
                facecolor='lightyellow', alpha=0.5)

    # 添加网格线
    plt.grid(True)

    # 取消外框
    plt.box(False)

    # 自动调整日期标签的格式和位置，以避免重叠
    plt.gcf().autofmt_xdate()
    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()
