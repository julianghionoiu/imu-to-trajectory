import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# === 1. Load the data ===

def load_sensor_data(filename, columns):
    df = pd.read_csv(filename, sep=r'\s+', header=0, names=columns)
    df['time'] = (df['TIMESTAMP'] - df['TIMESTAMP'].min()) * 1e-9  # normalize to seconds
    return df

acc_columns = ['TIMESTAMP', 'acc_x', 'acc_y', 'acc_z']
gyro_columns = ['TIMESTAMP', 'gyro_x', 'gyro_y', 'gyro_z']
mag_columns = ['TIMESTAMP', 'mag_x', 'mag_y', 'mag_z']

acc_df = load_sensor_data('samples/cal_2d_spin/ebay_table_spin/acc.data', acc_columns)
gyro_df = load_sensor_data('samples/cal_2d_spin/ebay_table_spin/gyro.data', gyro_columns)
mag_df = load_sensor_data('samples/cal_2d_spin/ebay_table_spin/mag.data', mag_columns)

# === 2. Define common time base from gyro timestamps ===

common_time = gyro_df['time'].values

# === 3. Interpolate accelerometer and magnetometer to gyro timestamps ===

def interpolate_to_common_time(df, time_col, data_cols, new_time):
    interpolated = {}
    for col in data_cols:
        interp_fn = interp1d(df[time_col], df[col], kind='linear', fill_value='extrapolate')
        interpolated[col] = interp_fn(new_time)
    return pd.DataFrame(interpolated)

acc_interp = interpolate_to_common_time(acc_df, 'time', ['acc_x', 'acc_y', 'acc_z'], common_time)
mag_interp = interpolate_to_common_time(mag_df, 'time', ['mag_x', 'mag_y', 'mag_z'], common_time)

# === 4. Merge synchronized data ===

synced_df = pd.DataFrame({
    'time': common_time,
    'gyro_x': gyro_df['gyro_x'].values,
    'gyro_y': gyro_df['gyro_y'].values,
    'gyro_z': gyro_df['gyro_z'].values,
    'acc_x': acc_interp['acc_x'],
    'acc_y': acc_interp['acc_y'],
    'acc_z': acc_interp['acc_z'],
    'mag_x': mag_interp['mag_x'],
    'mag_y': mag_interp['mag_y'],
    'mag_z': mag_interp['mag_z'],
})

# Optional: save or inspect
synced_df.to_csv('01_out_synced_imu_data.csv', index=False)
print(synced_df.head())

