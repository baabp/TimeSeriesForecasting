import os
import numpy as np
import pandas as pd
from zope.dottedname.resolve import resolve
from sklearn.model_selection import train_test_split

COL_DATE_DELHI = 'date'
COLS_TARGET_DELHI = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
COLS_TRANSFORM_DELHI = ['humidity', 'wind_speed', 'meanpressure']
COLS_LOG_DELHI = ['meantemp']

eps = 1e-11
NAN = np.NAN


def transform_log_ratios(vals):
    aux = np.log((vals[1:] + eps) / (vals[0:-1] + eps))
    return np.hstack(([NAN], aux))


def inverse_transform_log_ratios(log_ratio, temp_prev):
    return np.multiply(temp_prev, np.exp(log_ratio))


class TimeSeriesDatasetDelhi(object):
    def __init__(self, df_train, df_test=None, col_date=COL_DATE_DELHI, cols_target=COLS_TARGET_DELHI):
        self.df_train = df_train
        self.df_test = df_test
        self.col_date = col_date
        self.cols_target = cols_target

        self.df_val = None

        self.col_add_scale = 'scaled'
        self.col_add_log = 'log'

    # check null
    def remove_null_row(self):
        self.df_train = self.df_train[~self.df_train.isnull().any(axis=1)]
        self.df_test = self.df_test[~self.df_test.isnull().any(axis=1)]

    def _apply_scale(self, df, cols, scaler):
        transformed_train = scaler.transform(df.loc[:, cols])
        df_train_transformed = pd.DataFrame(transformed_train, columns=[col + '_' + self.col_add_scale for col in cols])
        return pd.concat([df, df_train_transformed], axis=1)

    def transform_cols(self, cols=COLS_TRANSFORM_DELHI, scaler_class='sklearn.preprocessing.MinMaxScaler',
                       replace=True):
        scaler = resolve(scaler_class)()
        scaler.fit(self.df_train.loc[:, cols])

        # update self.cols_target
        self.cols_target = [col + '_' + self.col_add_scale if col in cols else col for col in self.cols_target]

        df_train_transformed = self._apply_scale(self.df_train, cols=cols, scaler=scaler)
        if replace:
            self.df_train = df_train_transformed.copy()

        if self.df_test is not None:
            df_test_transformed = self._apply_scale(self.df_test, cols=cols, scaler=scaler)
            if replace:
                self.df_test = df_test_transformed.copy()
        else:
            df_test_transformed = None

        return df_train_transformed, df_test_transformed

    def log_cols(self, cols=COLS_LOG_DELHI):
        for col in cols:
            self.df_train[col + '_' + self.col_add_log] = transform_log_ratios(self.df_train[col].values)
            if self.df_test is not None:
                self.df_test[col + '_' + self.col_add_log] = transform_log_ratios(self.df_test[col].values)

        # update self.cols_target
        self.cols_target = [col + '_' + self.col_add_log if col in cols else col for col in self.cols_target]

        # remove null rows
        self.remove_null_row()

        return self.df_train, self.df_test

    def _get_look_back_data(self, df, look_back):
        df = df.reset_index(drop=True)
        x = []
        y = []
        for i in range(1, len(df) - look_back - 1):
            x.append(df.loc[i:i + look_back - 1, self.cols_target].values)
            y.append(df.loc[i + look_back, self.cols_target].values.tolist())
        X = np.array(x).reshape(-1, look_back, len(self.cols_target))
        y = np.array(y)
        list_datetime = [timestamp.to_pydatetime() for timestamp in df[self.col_date].to_list()]
        return X, y, list_datetime

    def get_ts_dataset(self, look_back=6):
        X_train, y_train, list_datetime_train = self._get_look_back_data(self.df_train, look_back=look_back)
        if self.df_test is not None:
            X_test, y_test, list_datetime_test = self._get_look_back_data(self.df_test, look_back=look_back)
        else:
            X_test = None
            y_test = None
            list_datetime_test = None
        return X_train, y_train, list_datetime_train, X_test, y_test, list_datetime_test


def main():
    path_data_train = 'data/DailyDelhiClimate/DailyDelhiClimateTrain.csv'
    path_data_test = 'data/DailyDelhiClimate/DailyDelhiClimateTest.csv'
    look_back = 6
    test_size = 0.2

    df_train = pd.read_csv(path_data_train)
    df_test = pd.read_csv(path_data_test)

    print(df_train.head())
    print(df_train.describe())

    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])

    time_series_dataset_delhi = TimeSeriesDatasetDelhi(df_train=df_train,
                                                       df_test=df_test)
    time_series_dataset_delhi.remove_null_row()
    time_series_dataset_delhi.transform_cols()
    time_series_dataset_delhi.log_cols()
    X_train, y_train, list_datetime_train, X_test, y_test, list_datetime_test = \
        time_series_dataset_delhi.get_ts_dataset(look_back=look_back)

    X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train, test_size=test_size, shuffle=False)

if __name__ == '__main__':
    main()
