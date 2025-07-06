import pandas as pd
import numpy as np
import json
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


# === Load Data ===
base_dir = "./devices/E0A8AD21/gyr_calibration/zrot/"
acc_df = pd.read_csv(base_dir+"acc.data", sep=r'\s+', names=["timestamp", "acc_x", "acc_y", "acc_z"], header=0)
gyro_df = pd.read_csv(base_dir+"gyr.data", sep=r'\s+', names=["timestamp", "gyro_x", "gyro_y", "gyro_z"], header=0)
mag_df = pd.read_csv(base_dir+"mag.data", sep=r'\s+', names=["timestamp", "mag_x", "mag_y", "mag_z"], header=0)

# Normalize timestamps
base_time = min(acc_df["timestamp"].min(), gyro_df["timestamp"].min(), mag_df["timestamp"].min())
for df in [acc_df, gyro_df, mag_df]:
    df["time"] = (df["timestamp"] - base_time) * 1e-9

# === Magnetometer Calibration ===
with open("./devices/E0A8AD21/mag_calibration_parameters.json", "r") as f:
    mag_cal = json.load(f)
bias = np.array(mag_cal["bias"])
scale_matrix = np.array(mag_cal["scale_matrix"])
scale_inv = np.linalg.inv(np.linalg.cholesky(scale_matrix))
mag_raw = mag_df[["mag_x", "mag_y", "mag_z"]].values
mag_corrected = (scale_inv @ (mag_raw - bias).T).T
mag_df[["mag_x", "mag_y", "mag_z"]] = mag_corrected

# === Interpolate to common time base (gyro timestamps) ===
def interpolate(df, time_col, data_cols, target_time):
    return pd.DataFrame({col: interp1d(df[time_col], df[col], kind='linear', fill_value='extrapolate')(target_time) for col in data_cols})

common_time = gyro_df["time"].values
acc = interpolate(acc_df, "time", ["acc_x", "acc_y", "acc_z"], common_time).values * 9.80665 / 1000.0  # mg to m/s^2
gyro = interpolate(gyro_df, "time", ["gyro_x", "gyro_y", "gyro_z"], common_time).values * np.pi / 180
mag = interpolate(mag_df, "time", ["mag_x", "mag_y", "mag_z"], common_time).values * 1e5  # Gauss to nT
mag_raw_vals = interpolate(mag_df.assign(mag_x=mag_raw[:,0], mag_y=mag_raw[:,1], mag_z=mag_raw[:,2]),
                           "time", ["mag_x", "mag_y", "mag_z"], common_time).values

dt = np.mean(np.diff(common_time))

# === EKF Setup ===
def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def quat_to_rotmat(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1-2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3),   2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),   1-2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),   2*(q2*q3 + q0*q1),   1-2*(q1**2 + q2**2)]
    ])

def integrate_gyro(q, gyro, dt):
    wx, wy, wz = gyro
    omega = np.array([
        [0.0, -wx, -wy, -wz],
        [wx,  0.0,  wz, -wy],
        [wy, -wz,  0.0, wx],
        [wz,  wy, -wx,  0.0]
    ])
    dq = 0.5 * omega @ q
    return normalize_quaternion(q + dq * dt)

# Initial orientation
def acc_mag_to_quaternion(acc, mag):
    # Normalize accelerometer and magnetometer
    g = acc / np.linalg.norm(acc)
    m = mag / np.linalg.norm(mag)

    # Calculate East and North vectors
    east = np.cross(m, g)
    east /= np.linalg.norm(east)

    north = np.cross(g, east)

    # Construct rotation matrix (column-wise: X=East, Y=North, Z=Up)
    R_mat = np.column_stack((east, north, g))

    # Convert to quaternion (returns [x, y, z, w])
    return R.from_matrix(R_mat).as_quat()

N = len(common_time)
q = np.roll(acc_mag_to_quaternion(acc[0], mag[0]), 1)  # roll w to front
P = np.eye(4) * 0.01
Q_k = np.eye(4) * 0.001
R_k = np.eye(3) * 0.1
g_ref = np.array([0.0, 0.0, 9.81])
quaternions = np.zeros((N, 4))
quaternions[0] = q

m_ref = np.array([1.0, 0.0, 0.0])  # adjust if needed
R_m = np.eye(3) * 0.3  # magnetometer measurement noise

# === EKF Loop ===
for t in range(1, N):
    # === Prediction ===
    q_pred = integrate_gyro(q, gyro[t], dt)
    P = P + Q_k

    # === Accelerometer update ===
    R_mat = quat_to_rotmat(q_pred)
    g_meas = R_mat.T @ g_ref
    y = acc[t] - g_meas

    # Jacobian H for accelerometer
    H = np.zeros((3, 4))
    eps = 1e-5
    for i in range(4):
        dq = np.zeros(4)
        dq[i] = eps
        q_pert = normalize_quaternion(q_pred + dq)
        R_pert = quat_to_rotmat(q_pert)
        g_pert = R_pert.T @ g_ref
        H[:, i] = (g_pert - g_meas) / eps

    S = H @ P @ H.T + R_k
    K = P @ H.T @ np.linalg.inv(S)
    dq = K @ y
    q = normalize_quaternion(q_pred + dq)
    P = (np.eye(4) - K @ H) @ P

    # === Magnetometer update ===
    m_meas = mag[t] / np.linalg.norm(mag[t])  # normalize measured field
    m_pred = quat_to_rotmat(q).T @ m_ref      # predicted field in body frame

    y_m = m_meas - m_pred

    # Jacobian H_m for magnetometer
    H_m = np.zeros((3, 4))
    for i in range(4):
        dq = np.zeros(4)
        dq[i] = eps
        q_pert = normalize_quaternion(q + dq)
        R_pert = quat_to_rotmat(q_pert)
        m_pert = R_pert.T @ m_ref
        H_m[:, i] = (m_pert - m_pred) / eps

    S_m = H_m @ P @ H_m.T + R_m
    K_m = P @ H_m.T @ np.linalg.inv(S_m)
    dq_m = K_m @ y_m

    q = normalize_quaternion(q + dq_m)
    P = (np.eye(4) - K_m @ H_m) @ P

    # Store result
    quaternions[t] = q

# === Post-processing ===
euler_angles = R.from_quat(quaternions).as_euler('xyz', degrees=True)
gravity_sensor = np.array([quat_to_rotmat(q).T @ g_ref for q in quaternions])
motion_acc = acc - gravity_sensor

# === Plotting ===
fig, axs = plt.subplots(6, 1, figsize=(12, 12), sharex=True)

axs[0].plot(common_time, gyro)
axs[0].set_ylabel('Gyro (rad/s)')
axs[0].legend(['X', 'Y', 'Z'])
axs[0].set_title('Gyroscope')

axs[1].plot(common_time, acc)
axs[1].set_ylabel('Accel (m/s²)')
axs[1].legend(['X', 'Y', 'Z'])
axs[1].set_title('Accelerometer')

axs[2].plot(common_time, mag)
axs[2].plot(common_time, mag_raw_vals, '--')
axs[2].set_ylabel('Mag (uT)')
axs[2].legend(['Cal X', 'Cal Y', 'Cal Z', 'Raw X', 'Raw Y', 'Raw Z'])
axs[2].set_title('Magnetometer (Calibrated vs Raw)')

axs[3].plot(common_time, motion_acc)
axs[3].set_ylabel('Motion Acc (m/s²)')
axs[3].legend(['X', 'Y', 'Z'])
axs[3].set_title('Gravity Removed Acceleration')

axs[4].plot(common_time, euler_angles)
axs[4].set_ylabel('Angle wrapped (deg)')
axs[4].legend(['Roll', 'Pitch', 'Yaw'])
axs[4].set_title('Euler Angles (EKF)')

euler_unwrapped = np.unwrap(np.radians(euler_angles), axis=0)
euler_unwrapped_deg = np.degrees(euler_unwrapped)
axs[5].plot(common_time, euler_unwrapped_deg)
axs[5].set_ylabel('Angle unwrapped (deg)')
axs[5].legend(['Roll', 'Pitch', 'Yaw'])
axs[5].set_title('Euler Angles (EKF)')

plt.grid(True)
plt.tight_layout()
plt.show()
