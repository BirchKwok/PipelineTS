import re
import pandas as pd
import os

import yaml
from spinesUtils.asserts import raise_if_not
from tabulate import tabulate

from ..frame import DataTS

FILE_PATH = os.path.dirname(__file__)


def _call_name(built_in_func, name):
    """Wrapper of built-in datasets

    Parameters
    ----------
    built_in_func : BuiltInSeriesData class
    name : dataset's name

    Returns
    -------
    wrapped func
    """
    raise_if_not(ValueError, name is not None, "name must be a string")

    def wrap():
        return built_in_func[name]

    return wrap


class DataWrapper(DataTS):
    def __init__(self, dataset, name):
        super().__init__(dataset, name)
        self.fp = os.path.join(FILE_PATH, './built-in-datasets/metadata.yaml')
        self.name = name

    @property
    def target_col(self):
        with open(self.fp, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)[self.name][0]['target_col']

    @property
    def feature_cols(self):
        with open(self.fp, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)[self.name][0]['feature_cols']

    @property
    def time_col(self):
        with open(self.fp, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)[self.name][0]['time_col']


class BuiltInSeriesData:
    """Load the built-in data

    Parameters
    ----------
    print_file_list : bool, whether to print the exists file name list

    Returns
    -------
    None
    """

    def __init__(self, print_file_list=True):
        self.file_list = sorted(os.listdir(os.path.join(FILE_PATH, './built-in-datasets/')))
        if print_file_list:
            print(self)

    def _load_data(self, fp):
        self._FILEPATH = os.path.join(FILE_PATH, './built-in-datasets/', fp)
        if not os.path.exists(self._FILEPATH):
            self._FILEPATH = fp

        raise_if_not(FileNotFoundError, os.path.exists(self._FILEPATH), f'No such file or directory: {self._FILEPATH}')

        return DataWrapper(pd.read_csv(self._FILEPATH, sep=','), name='.'.join(fp.split('.')[:-1]))

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._load_data(self.file_list[item])

        elif isinstance(item, str):
            if not item.endswith('.csv'):
                item = item + '.csv'
            return self._load_data(self.file_list[self.file_list.index(item)])
        else:
            raise KeyError(f"invalid key: {item}")

    def __len__(self):
        return len(self.file_list)

    @property
    def names(self):
        """Returns the built-in series data names-list."""
        return [''.join(i.split('.')[:-1]) for i in self.file_list]

    def __str__(self):
        table = []
        for i in range(len(self.file_list)):
            _ = [re.split('\.', self.file_list[i])[0].strip(),
                 ', '.join(self[i].columns.tolist())]
            table.append(_)

        return tabulate(table, headers=["ds name", "columns"], showindex="always",
                        tablefmt="pretty", colalign=("right", "left", "left"))

    def __repr__(self):
        return self.__str__()


LoadElectricDataSets = _call_name(BuiltInSeriesData(print_file_list=False), 'Electric_Production')
LoadMessagesSentDataSets = _call_name(BuiltInSeriesData(print_file_list=False), 'Messages_Sent')
LoadMessagesSentHourDataSets = _call_name(BuiltInSeriesData(print_file_list=False), 'Messages_Sent_Hour')
LoadWebSales = _call_name(BuiltInSeriesData(print_file_list=False), 'Web_Sales')
LoadSupermarketIncoming = _call_name(BuiltInSeriesData(print_file_list=False), 'Supermarket_Incoming')
