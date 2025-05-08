from sys import platform
from typing import Iterable, Iterator, List
import numpy as np
import ctypes
import pandas as pd
from os import path

# C struct binding representing a loaded data point from a thermite log
class _datapoint_t(ctypes.Structure):
    _fields_ = [("timestamp", ctypes.c_int64),
                ("value", ctypes.c_double)]

# C struct binding representing a loaded header from a thermite log
class _header_t(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char * 48),
                ("start", ctypes.c_uint64)]

# Load thermite dynamic lib
if platform == "win32":
    _libthermite = ctypes.cdll.LoadLibrary(path.join(path.dirname(__file__), "thermite.dll"))
elif platform == "darwin":
    _libthermite = ctypes.cdll.LoadLibrary(path.join(path.dirname(__file__), "libthermite.dylib"))


class Thermite:
    def __init__(self, path: str):
        """Load a thermite file
        Parameters
        ----------
        ``path``: File path to a thermite file. This will crash if the file path does not refer to a valid thermite file
        """
        self.__path = path
        self.__headers = []
        self.__data = dict()
        self.__load_headers()

    def __cstr_path(self):
        return self.__path.encode("utf-8")

    def __load_headers(self):
        header_count = _libthermite.thermite_header_count(self.__cstr_path())
        if header_count < 0:
            raise Exception(f"Error when loading thermite file. Error code {header_count}")
        print(header_count)

        header_buf = (_header_t * header_count)()
        err = _libthermite.thermite_headers(self.__cstr_path(), ctypes.byref(header_buf), header_count)
        if err < 0:
            raise Exception(f"Error when loading thermite file. Error code {header_count}")
        for header in header_buf:
            self.__headers.append(header.name.decode("utf-8"))

    def __getitem__(self, name: str) -> np.ndarray:
        """Load a signal from the thermite file
        Parameters
        ----------
        ``name``: An iterable list of thermite signal names to pull

        Returns
        -------
        numpy array containing an ordered list of (timestamp, value) tuples. Timestamps are stored as microseconds since unix epoch (Jan 1, 1970).
        """
        if name not in self.__headers:
            return np.array([])
        if name in self.__data:
            return self.__data[name]
        bname = name.encode("utf-8")
        data_count = _libthermite.thermite_data_count(self.__cstr_path(), bname)
        if data_count < 0:
            raise Exception(f"Error when loading signal {name} from thermite file. Error code {data_count}")

        data_buf = (_datapoint_t * data_count)()
        err = _libthermite.thermite_data(self.__cstr_path(), bname, ctypes.byref(data_buf), data_count)
        if err < 0:
            raise Exception(f"Error when loading signal {name} from thermite file. Error code {err}")

        python_result = []
        for datapoint in data_buf:
            python_result.append((datapoint.timestamp, datapoint.value))

        python_result = np.array(python_result)

        self.__data[name] = python_result
        return python_result

    def signals(self) -> List[str]:
        """Returns a list containing every signal name present in the thermite file"""
        return self.__headers

    def clear_cache(self):
        """When data is loaded from a thermite file, it gets cached so it can be quickly retrieved again. If you're loading a ton of thermite data however, this cache may become exceedingly large. If memory usage becomes a problem, clearing this cache may help significantly"""
        self.__data = dict()

    def load_df(self, names: Iterable[str], ffill=False, relative_timestamp=True) -> pd.DataFrame:
        """Load a Pandas DataFrame from Thermite. The index of the Dataframe is the time in seconds since the start of the first signal in the Dataframe
        Parameters
        ----------
        ``names``: An iterable list of thermite signal names to pull
        ``ffill``: Fill empty cells with previous value. By default this is disabled.
        ``relative_timestamp``: Converts timestamps into seconds since the timestamp of the earliest data point in the table. Disabling this will represent timestamps as seconds since unix epoch (Jan 1, 1970). By default, this is enabled

        Returns
        -------
        DataFrame containing requested thermite data if at least one requested exists in the thermite file. Otherwise, None is returned.
        """
        df = pd.DataFrame()
        for name in names:
            local_df = pd.DataFrame()
            raw = self[name]
            if len(raw) > 0:
                local_df["time"] = raw.T[0] / 1e6
                local_df[name] = raw.T[1]
                local_df.set_index("time", inplace=True)
                df = df.join(local_df, how='outer')

        if df.empty:
            return df
        if relative_timestamp:
            df.set_index(df.index - df.first_valid_index(), inplace=True)
        if ffill:
            df.ffill(inplace=True)
        return df

    def __contains__(self, name: str):
        return name in self.__headers

