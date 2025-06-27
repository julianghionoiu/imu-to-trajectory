import pandas as pd
import numpy as np
from ahrs.filters import Madgwick
from ahrs.common.orientation import q2euler

# === 1. Load synchronized IMU data ===
df = pd.read_csv("01_out_synced_imu_data.csv")

# === 2. Prepare input arrays ===
# Convert gyroscope from deg/s to rad/s
gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].to_numpy() * np.pi / 180.0
acc = df[['acc_x', 'acc_y', 'acc_z']].to_numpy()
mag = df[['mag_x', 'mag_y', 'mag_z']].to_numpy()

# Estimate sampling rate (assumes uniform sampling)
dt = np.mean(np.diff(df['time'].values))

# === 3. Initialize and run Madgwick filter ===
madgwick = Madgwick(sampleperiod=dt)
quaternions = np.zeros((len(df), 4))
q = np.array([1.0, 0.0, 0.0, 0.0])  # Initial orientation

for i in range(len(df)):
    q = madgwick.updateMARG(q, gyr=gyro[i], acc=acc[i], mag=mag[i])
    quaternions[i] = q

# === Save quaternions
quat_df = pd.DataFrame(quaternions, columns=["w", "x", "y", "z"])
quat_df["time"] = df["time"].values  # align time from original synced data
quat_df.to_csv("02_out_orientation_quaternions.csv", index=False)
print("Saved quaternions to orientation_quaternions.csv")


# Convert quaternions to Euler angles (roll, pitch, yaw)
euler_rad = np.array([q2euler(q) for q in quaternions])  # shape: (N, 3)
euler_deg = np.degrees(euler_rad)

# Create DataFrame
euler_df = pd.DataFrame(euler_deg, columns=['roll_deg', 'pitch_deg', 'yaw_deg'])
euler_df['time'] = df['time']

# === 5. Save euler angles ===
euler_df.to_csv("02_out_orientation_output.csv", index=False)
print(euler_df.head())
