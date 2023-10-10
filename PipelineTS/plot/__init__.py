import pandas as pd

from spinesUtils.asserts import ParameterTypeAssert, ParameterValuesAssert


@ParameterTypeAssert({
    'data': pd.DataFrame,
    'test_data': pd.DataFrame,
    'time_col': str,
    'target_col': str,
    'labels': (None, list, tuple)
})
@ParameterValuesAssert({
    'labels': lambda s: (s is None) or (len(s) == 2 and all(isinstance(i, str) for i in s))
})
def plot_data_period(data, test_data, time_col, target_col, labels=None):
    """可视化日期数据"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as m_dates

    if labels is None:
        labels = ('Data', 'Test Data')
    # 绘制曲线
    plt.plot_date(data[time_col], data[target_col], label=labels[0],
                  color='black', linestyle='-', marker='')
    plt.plot_date(test_data[time_col], test_data[target_col],
                  label=labels[1], color='blue', linestyle='-', marker='')

    # 添加竖线
    plt.axvline(x=test_data[time_col].iloc[0], color='black', linestyle='--')

    valid_data_max = test_data[target_col].max()
    valid_period = [test_data[time_col].iloc[0], test_data[time_col].iloc[-1]]

    plt.axvspan(valid_period[0], valid_period[1], ymin=0, ymax=valid_data_max,
                facecolor='lightyellow', alpha=0.5)

    # 设置x轴为日期格式
    date_fmt = m_dates.DateFormatter('%Y-%m-%d')
    plt.gca().xaxis.set_major_formatter(date_fmt)

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
