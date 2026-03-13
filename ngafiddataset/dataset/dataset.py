import gdown
import pandas as pd
import numpy as np
from sklearn import preprocessing
from tqdm.autonotebook import tqdm
import tensorflow as tf
from loguru import logger
from ngafiddataset.utils import shell_exec
import os
import tarfile
import shutil
import typing
from compress_pickle import load

from ngafiddataset.dataset.utils import *


def linear_interpolate_nan(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    arr = arr.astype(np.float32)

    T, C = arr.shape

    for c in range(C):
        channel = arr[:, c]
        nan_mask = np.isnan(channel)

        if not nan_mask.any():
            continue

        valid_idx = np.where(~nan_mask)[0]

        if len(valid_idx) == 0:
            arr[:, c] = 0.0
            continue

        arr[:, c] = np.interp(
            np.arange(T),
            valid_idx,
            channel[valid_idx]
        )

    return arr


class NGAFID_Dataset_Downloader:

    # Change "2days" Google Drive link
    ngafid_urls = {
        "all_flights": "https://drive.google.com/uc?id=1-0pVPhwRQoifT_VuQyGDLXuzYPYySX-Y",
        # "2days": "https://drive.google.com/uc?id=1-2pxwiQNhFnhTg7whosQoF_yztD5jOM2",
        "2days": "https://drive.google.com/uc?id=1KIQKQOu9oMed_RMxtwn3Zpc2nC21viT_",
    }

    @classmethod
    def download(cls, name: str, destination: str = '', extract=True):

        assert name in cls.ngafid_urls.keys()

        url = cls.ngafid_urls[name]
        output = os.path.join(destination, "%s.tar.gz" % name)

        if not os.path.exists(output):
            gdown.download(url, output, quiet=False)

        if extract:
            logger.info('Extracting File')
            _ = tarfile.open(output).extractall(destination)

        return name, destination


class NGAFID_Dataset_Manager(NGAFID_Dataset_Downloader):

    channels = 23

    def __init__(
        self,
        name: str,
        destination: str = '',
        max_length=4096,
        extract=True,
        nan_fill_method='none',
        use_sliding_window=False,
        window_stride=2048,
        **kwargs
    ):
        assert name in self.ngafid_urls.keys()

        if name == 'all_flights':
            logger.info(
                'Downloading and extracting Parquet Files to %s\\one_parq. Please open them using dask dataframes'
                % destination
            )
            self.download(name, destination, extract=True)

        else:
            self.name = name
            self.max_length = max_length
            self.destination = destination


            self.nan_fill_method = nan_fill_method
            self.use_sliding_window = use_sliding_window
            self.window_stride = window_stride

            self.files = ['flight_data.pkl', 'flight_header.csv', 'stats.csv']
            self.files = {file: os.path.join(destination, name, file) for file in self.files}

            # self.download(name, destination, extract)

            self.flight_header_df = pd.read_csv(self.files['flight_header.csv'], index_col='Master Index')
            self.flight_data_array = load(self.files['flight_data.pkl'])
            self.flight_stats_df = pd.read_csv(self.files['stats.csv'])

            # self.data_dict = self.construct_data_dictionary()

            self.maxs = self.flight_stats_df.iloc[0, 1:24].to_numpy(dtype=np.float32)
            self.mins = self.flight_stats_df.iloc[1, 1:24].to_numpy(dtype=np.float32)

    def construct_data_dictionary(
        self,
        numpy=True,
        stride=None,
        nan_fill_method=None,
        use_sliding_window=None
    ):
        data_dict = []

        if stride is None:
            stride = self.window_stride
        if nan_fill_method is None:
            nan_fill_method = self.nan_fill_method
        if use_sliding_window is None:
            use_sliding_window = self.use_sliding_window

        for index, row in tqdm(self.flight_header_df.iterrows(), total=len(self.flight_header_df)):

            window_id = 0
            flight = self.flight_data_array[index]

            # =========================================================
            # Optional NaN fill
            #   - 'linear' / True : Linear Interpolation
            #   - 'none' / None / False : replace_nan_w_zero
            # =========================================================
            if nan_fill_method in ['linear', True]:
                flight = linear_interpolate_nan(flight)
            elif nan_fill_method in ['none', None, False]:
                pass
            else:
                raise ValueError(f"Unsupported nan_fill_method: {nan_fill_method}")

            T = flight.shape[0]

            # =========================================================
            # no sliding window -> truncate to last max_length and zero-pad
            # =========================================================
            if not use_sliding_window:

                arr = np.zeros((self.max_length, self.channels), dtype=np.float16)

                to_pad = flight[-self.max_length:, :]

                arr[:to_pad.shape[0], :] += to_pad

                if not numpy:
                    arr = tf.convert_to_tensor(arr, dtype=tf.bfloat16)

                data_dict.append({
                    'id': index,
                    'window_id': window_id,
                    'data': arr,
                    'class': row['class'],
                    'fold': row['fold'],
                    'target_class': row['target_class'],
                    'before_after': row['before_after'],
                    'hclass': row['hclass']
                })

            # =========================================================
            # Sliding window
            # =========================================================
            else:

                if T <= self.max_length:

                    arr = np.zeros((self.max_length, self.channels), dtype=np.float16)
                    arr[:T, :] = flight

                    if not numpy:
                        arr = tf.convert_to_tensor(arr, dtype=tf.bfloat16)

                    data_dict.append({
                        'id': index,
                        'window_id': window_id,
                        'data': arr,
                        'class': row['class'],
                        'fold': row['fold'],
                        'target_class': row['target_class'],
                        'before_after': row['before_after'],
                        'hclass': row['hclass']
                    })

                else:

                    starts = list(range(0, T - self.max_length + 1, stride))

                    if starts[-1] != T - self.max_length:
                        starts.append(T - self.max_length)

                    for start in starts:

                        window = flight[start:start + self.max_length]
                        arr = window.astype(np.float16)

                        if not numpy:
                            arr = tf.convert_to_tensor(arr, dtype=tf.bfloat16)

                        data_dict.append({
                            'id': index,
                            'window_id': window_id,
                            'data': arr,
                            'class': row['class'],
                            'fold': row['fold'],
                            'target_class': row['target_class'],
                            'before_after': row['before_after'],
                            'hclass': row['hclass']
                        })

                        window_id += 1

        return data_dict

    def get_tf_dataset(
        self,
        fold=0,
        training=False,
        shuffle=False,
        batch_size=64,
        repeat=False,
        mode='before_after',
        ds=None
    ):
        if ds is None:
            ds = tf.data.Dataset.from_tensor_slices(
                to_dict_of_list(
                    get_slice(self.data_dict, fold=fold, reverse=training)
                )
            )

        ds = ds.repeat() if repeat else ds
        ds = ds.shuffle(shuffle) if shuffle else ds

        # min-max scaling
        ds = ds.map(get_dict_mod('data', get_scaler(self.maxs, self.mins)))

        ds = ds.map(get_dict_mod('data', replace_nan_w_zero))

        ds = ds.map(get_dict_mod('data', lambda x: tf.cast(x, tf.float32)))

        if mode == 'before_after':
            ds = ds.map(lambda x: (x['data'], x['before_after']))
        elif mode == 'classes':
            ds = ds.map(lambda x: (x['data'], x['target_class']))
        elif mode == 'both':
            ds = ds.map(lambda x: (
                {'data': x['data']},
                {'before_after': x['before_after'], 'target_class': x['target_class']}
            ))
        elif mode == 'hierarchy_basic':
            ds = ds.map(lambda x: (
                {'data': x['data']},
                {'before_after': x['before_after'], 'target_class': x['hclass']}
            ))
        else:
            raise KeyError

        ds = ds.batch(batch_size, drop_remainder=True) if batch_size else ds

        return ds

    def get_numpy_dataset(self, fold=0, training=False):
        return to_dict_of_list(get_slice(self.data_dict, fold=fold, reverse=training))