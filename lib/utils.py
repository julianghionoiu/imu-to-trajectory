import json

import numpy as np
import pandas as pd
from ahrs.common.orientation import q2R
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

import matplotlib

from lib.mag import compute_calibration_parameters, calibrate_np

matplotlib.use('TkAgg')

def calibrate_mag_from_data(mag_df):
    bias, scale_matrix = compute_calibration_parameters(mag_df)
    return calibrate_np(mag_df, bias, scale_matrix)

def calibrate_mag_from_file(mag_df):
    # === Load Magnetometer Calibration ===
    with open("./devices/E0A8AD21/mag_calibration_parameters.json", "r") as f:
        mag_cal = json.load(f)
    bias = np.array(mag_cal["bias"])
    scale_matrix = np.array(mag_cal["scale_matrix"])
    scale_inv = np.linalg.inv(np.linalg.cholesky(scale_matrix))
    # Apply calibration to MAG
    mag_raw = mag_df[["mag_x", "mag_y", "mag_z"]].values
    mag_corrected = (scale_inv @ (mag_raw - bias).T).T
    mag_df_copy = mag_df.copy()
    mag_df_copy[["mag_x", "mag_y", "mag_z"]] = mag_corrected
    return mag_df_copy
    

def load_from_dir(base_dir=None):
    global common_time, gyro, acc, mag, dt
    acc_df = pd.read_csv(base_dir + "acc.data", sep=r'\s+', names=["timestamp", "acc_x", "acc_y", "acc_z"], header=0)
    gyro_df = pd.read_csv(base_dir + "gyr.data", sep=r'\s+', names=["timestamp", "gyro_x", "gyro_y", "gyro_z"],
                          header=0)
    mag_df = pd.read_csv(base_dir + "mag.data", sep=r'\s+', names=["timestamp", "mag_x", "mag_y", "mag_z"], header=0)
    # Normalize timestamps to seconds since start
    base_time = min(acc_df["timestamp"].min(), gyro_df["timestamp"].min(), mag_df["timestamp"].min())
    acc_df["time"] = (acc_df["timestamp"] - base_time) * 1e-9
    gyro_df["time"] = (gyro_df["timestamp"] - base_time) * 1e-9
    mag_df["time"] = (mag_df["timestamp"] - base_time) * 1e-9

    mag_df = calibrate_mag_from_file(mag_df)
    # === Interpolate all data to gyro timestamps ===
    common_time = gyro_df["time"].values

    def interpolate(df, time_col, data_cols, target_time):
        interp_data = {}
        for col in data_cols:
            f = interp1d(df[time_col], df[col], kind='linear', bounds_error=False, fill_value="extrapolate")
            interp_data[col] = f(target_time)
        return pd.DataFrame(interp_data)

    gyro = gyro_df[["gyro_x", "gyro_y", "gyro_z"]].values
    acc_interp = interpolate(acc_df, "time", ["acc_x", "acc_y", "acc_z"], common_time).values
    mag_interp = interpolate(mag_df, "time", ["mag_x", "mag_y", "mag_z"], common_time).values
    return common_time, mag_interp, acc_interp, gyro

def gyr_convert_deg_to_rads(values):
    return np.radians(values)

def acc_convert_mg_to_mps2(values):
    return values * 9.80665 / 1000.0  # mg to m/s^2

def mag_convert_gauss_to_nt(values):
    return values * 1e5  # Gauss to nT

def mag_convert_gauss_to_mt(values):
    return values * 0.1  # Gauss to mT

def frequency_from_time(common_time=None):
    dt = np.diff(common_time)
    return 1.0 / np.mean(dt)

def remove_gravity(common_time=None, acc=None, quaternions=None):
    gravity_sensor = np.zeros_like(acc)
    for i in range(len(common_time)):
        Rm = q2R(quaternions[i])  # Converts quaternion to rotation matrix
        # g = Rm.T @ np.array([0, 0, 9.81])  # gravity vector in sensor frame
        g = Rm.T @ np.array([0, 0, 9.81])  # gravity vector in sensor frame
        gravity_sensor[i] = g
    return acc - gravity_sensor

def plot_imu_data(common_time=None, gyro=None, acc=None, mag=None, motion_acc=None, quaternions=None):
    # === Convert quaternions to Euler angles (roll, pitch, yaw) ===
    # The 'xyz' order corresponds to roll, pitch, yaw (in radians)
    euler_angles = R.from_quat(quaternions).as_euler('xyz', degrees=True)  # shape: (N, 3)
    # euler_angles[:, 0] = roll, euler_angles[:, 1] = pitch, euler_angles[:, 2] = yaw
    
    # === Plot IMU Data: Gyro, Acc, Mag (with raw mag) ===
    fig, axs = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
    # Gyroscope
    axs[0].plot(common_time, gyro[:, 0], label='Gyro X')
    axs[0].plot(common_time, gyro[:, 1], label='Gyro Y')
    axs[0].plot(common_time, gyro[:, 2], label='Gyro Z')
    axs[0].set_ylabel('Gyro (rad/s)')
    axs[0].legend()
    axs[0].set_title('Gyroscope Data')
    # Accelerometer
    axs[1].plot(common_time, acc[:, 0], label='Acc X')
    axs[1].plot(common_time, acc[:, 1], label='Acc Y')
    axs[1].plot(common_time, acc[:, 2], label='Acc Z')
    axs[1].set_ylabel('Acceleration (m/s²)')
    axs[1].legend()
    axs[1].set_title('Accelerometer Data')
    # Magnetometer (cal)
    axs[2].plot(common_time, mag[:, 0], label='Mag X (cal)')
    axs[2].plot(common_time, mag[:, 1], label='Mag Y (cal)')
    axs[2].plot(common_time, mag[:, 2], label='Mag Z (cal)')
    axs[2].set_ylabel('Magnetic Field (uT)')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()
    axs[2].set_title('Magnetometer Data (Calibrated)')
    # Motion Acceleration (gravity removed)
    axs[3].plot(common_time, motion_acc[:, 0], label='Motion Acc X')
    axs[3].plot(common_time, motion_acc[:, 1], label='Motion Acc Y')
    axs[3].plot(common_time, motion_acc[:, 2], label='Motion Acc Z')
    axs[3].set_ylabel('Motion Acc (m/s²)')
    axs[3].legend()
    axs[3].set_title('Linear Acceleration (Gravity Removed)')
    axs[4].plot(common_time, euler_angles)
    axs[4].set_ylabel('Angle wrapped (deg)')
    axs[4].legend(['Roll', 'Pitch', 'Yaw'])
    axs[4].set_title('Wrapped Euler Angles')
    euler_unwrapped = np.unwrap(np.radians(euler_angles), axis=0)
    euler_unwrapped_deg = np.degrees(euler_unwrapped)
    axs[5].plot(common_time, euler_unwrapped_deg)
    axs[5].set_ylabel('Angle unwrapped (deg)')
    axs[5].legend(['Roll', 'Pitch', 'Yaw'])
    axs[5].set_title('Unwrapped Euler Angles')
    plt.grid(True)
    plt.show()
    plt.tight_layout()
    plt.show()
