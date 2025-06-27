import pandas as pd
import numpy as np
import json
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick
from ahrs.common.orientation import q2R
from scipy.spatial.transform import Rotation as R

# === Load Data ===
acc_df = pd.read_csv("./devices/E0A8AD21/sensor_cal_acc.data", sep=r'\s+', names=["timestamp", "acc_x", "acc_y", "acc_z"], header=0)
gyro_df = pd.read_csv("./devices/E0A8AD21/sensor_cal_gyr.data", sep=r'\s+', names=["timestamp", "gyro_x", "gyro_y", "gyro_z"], header=0)
mag_df = pd.read_csv("./devices/E0A8AD21/sensor_cal_mag.data", sep=r'\s+', names=["timestamp", "mag_x", "mag_y", "mag_z"], header=0)

# Normalize timestamps to seconds since start
base_time = min(acc_df["timestamp"].min(), gyro_df["timestamp"].min(), mag_df["timestamp"].min())
acc_df["time"] = (acc_df["timestamp"] - base_time) * 1e-9
gyro_df["time"] = (gyro_df["timestamp"] - base_time) * 1e-9
mag_df["time"] = (mag_df["timestamp"] - base_time) * 1e-9

# === Load Magnetometer Calibration ===
with open("./devices/E0A8AD21/mag_calibration_parameters.json", "r") as f:
    mag_cal = json.load(f)

bias = np.array(mag_cal["bias"])
scale_matrix = np.array(mag_cal["scale_matrix"])
scale_inv = np.linalg.inv(np.linalg.cholesky(scale_matrix))

# Apply calibration to MAG
mag_raw = mag_df[["mag_x", "mag_y", "mag_z"]].values
mag_corrected = (scale_inv @ (mag_raw - bias).T).T
mag_df[["mag_x", "mag_y", "mag_z"]] = mag_corrected

# === Interpolate all data to gyro timestamps ===
common_time = gyro_df["time"].values

def interpolate(df, time_col, data_cols, target_time):
    interp_data = {}
    for col in data_cols:
        f = interp1d(df[time_col], df[col], kind='linear', bounds_error=False, fill_value="extrapolate")
        interp_data[col] = f(target_time)
    return pd.DataFrame(interp_data)

acc_interp = interpolate(acc_df, "time", ["acc_x", "acc_y", "acc_z"], common_time)
mag_interp = interpolate(mag_df, "time", ["mag_x", "mag_y", "mag_z"], common_time)

# === Prepare inputs for Madgwick ===
gyro = gyro_df[["gyro_x", "gyro_y", "gyro_z"]].values * np.pi / 180.0  # deg/s to rad/s
acc = acc_interp[["acc_x", "acc_y", "acc_z"]].values
mag = mag_interp[["mag_x", "mag_y", "mag_z"]].values
dt = np.mean(np.diff(common_time))

# === Run Madgwick filter ===
# Initialize filter
madgwick = Madgwick(sampleperiod=dt)

def acc_mag_to_quaternion(acc, mag):
    """
    Estimate initial orientation quaternion from accelerometer and magnetometer.
    acc: (3,) array, accelerometer reading (m/s^2)
    mag: (3,) array, magnetometer reading (uT)
    Returns: (4,) array, quaternion [x, y, z, w] (scipy format)
    """
    # Normalize accelerometer (gravity direction)
    g = acc / np.linalg.norm(acc)
    # Normalize magnetometer
    m = mag / np.linalg.norm(mag)

    # East vector (y axis): e = (m x g)
    e = np.cross(m, g)
    e /= np.linalg.norm(e)
    # North vector (x axis): n = (g x e)
    n = np.cross(g, e)
    n /= np.linalg.norm(n)
    # Down vector (z axis): -g

    # Rotation matrix: columns are [n, e, -g]
    R_mat = np.column_stack((n, e, -g))
    # Convert to quaternion (scipy expects [x, y, z, w])
    quat = R.from_matrix(R_mat).as_quat()
    return quat


# Allocate array for quaternions
quaternions = np.zeros((len(common_time), 4))
# --- With this: ---
initial_acc = acc[0]
initial_mag = mag[0]
quaternions[0] = acc_mag_to_quaternion(initial_acc, initial_mag)

# Run filter
for t in range(1, len(common_time)):
    quaternions[t] = madgwick.updateMARG(quaternions[t-1], gyr=gyro[t], acc=acc[t], mag=mag[t])

# === Convert quaternions to Euler angles (roll, pitch, yaw) ===
# The 'xyz' order corresponds to roll, pitch, yaw (in radians)
euler_angles = R.from_quat(quaternions).as_euler('xyz', degrees=True)  # shape: (N, 3)
# euler_angles[:, 0] = roll, euler_angles[:, 1] = pitch, euler_angles[:, 2] = yaw

# === Remove Gravity
gravity_sensor = np.zeros_like(acc)

for i in range(len(common_time)):
    R = q2R(quaternions[i])  # Converts quaternion to rotation matrix
    g = R.T @ np.array([0, 0, 9.81])  # gravity vector in sensor frame
    gravity_sensor[i] = g

motion_acc = acc - gravity_sensor


# === Interpolate raw magnetometer data to common time ===
mag_raw_interp = interpolate(mag_df.assign(mag_x=mag_raw[:,0], mag_y=mag_raw[:,1], mag_z=mag_raw[:,2]), 
                             "time", ["mag_x", "mag_y", "mag_z"], common_time)
mag_raw_vals = mag_raw_interp[["mag_x", "mag_y", "mag_z"]].values

# === Plot IMU Data: Gyro, Acc, Mag (with raw mag) ===
fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

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

# Magnetometer (raw and calibrated)
axs[2].plot(common_time, mag[:, 0], label='Mag X (calibrated)')
axs[2].plot(common_time, mag[:, 1], label='Mag Y (calibrated)')
axs[2].plot(common_time, mag[:, 2], label='Mag Z (calibrated)')
axs[2].plot(common_time, mag_raw_vals[:, 0], '--', label='Mag X (raw)')
axs[2].plot(common_time, mag_raw_vals[:, 1], '--', label='Mag Y (raw)')
axs[2].plot(common_time, mag_raw_vals[:, 2], '--', label='Mag Z (raw)')
axs[2].set_ylabel('Magnetic Field (uT)')
axs[2].set_xlabel('Time (s)')
axs[2].legend()
axs[2].set_title('Magnetometer Data (Raw vs Calibrated)')

# Motion Acceleration (gravity removed)
axs[3].plot(common_time, motion_acc[:, 0], label='Motion Acc X')
axs[3].plot(common_time, motion_acc[:, 1], label='Motion Acc Y')
axs[3].plot(common_time, motion_acc[:, 2], label='Motion Acc Z')
axs[3].set_ylabel('Motion Acc (m/s²)')
axs[3].legend()
axs[3].set_title('Linear Acceleration (Gravity Removed)')

# Euler Angles
axs[4].plot(common_time, euler_angles[:, 0], label='Roll')
axs[4].plot(common_time, euler_angles[:, 1], label='Pitch')
axs[4].plot(common_time, euler_angles[:, 2], label='Yaw')
axs[4].set_ylabel('Angle (deg)')
axs[4].legend()
axs[4].set_title('Euler Angles (Roll, Pitch, Yaw)')

plt.grid(True)
plt.show()

plt.tight_layout()
plt.show()