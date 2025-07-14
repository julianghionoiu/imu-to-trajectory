from scipy.spatial.transform import Rotation as R

import numpy as np

from lib.utils import load_from_dir, plot_imu_data, gyr_convert_deg_to_rads, acc_convert_mg_to_mps2, \
    mag_convert_gauss_to_nt

common_time, mag, acc, gyro = load_from_dir("./devices/E0A8AD21/gyr_calibration/zrot/")

gyro = gyr_convert_deg_to_rads(gyro)
acc = acc_convert_mg_to_mps2(acc)
mag = mag_convert_gauss_to_nt(mag)

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

# === EKF Loop ===
for t in range(1, N):
    q_pred = integrate_gyro(q, gyro[t], dt)
    P = P + Q_k

    R_mat = quat_to_rotmat(q_pred)
    g_meas = R_mat.T @ g_ref
    y = acc[t] - g_meas

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
    quaternions[t] = q

# === Post-processing ===
euler_angles = R.from_quat(quaternions).as_euler('xyz', degrees=True)
gravity_sensor = np.array([quat_to_rotmat(q).T @ g_ref for q in quaternions])
motion_acc = acc - gravity_sensor

plot_imu_data(common_time, gyro, acc, mag, motion_acc, quaternions)