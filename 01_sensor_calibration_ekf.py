from ahrs.filters import EKF

from lib.utils import remove_gravity, load_from_dir, frequency_from_time, plot_imu_data, gyr_convert_deg_to_rads, \
    acc_convert_mg_to_mps2, mag_convert_gauss_to_mt


common_time, mag, acc, gyro = load_from_dir("./devices/E0A8AD21/gyr_calibration/zrot/")

gyro = gyr_convert_deg_to_rads(gyro)
acc = acc_convert_mg_to_mps2(acc)
mag = mag_convert_gauss_to_mt(mag)

# default noises [0.3**2, 0.5**2, 0.8**2]
filter = EKF(gyr=gyro, acc=acc, mag=mag, frequency=frequency_from_time(common_time),
             frame='ENU',
             magnetic_ref=66,
             var_gyr=0.5**2,
             var_acc=0.1**2,
             var_mag=0.8**2
             )

quaternions = filter.Q

motion_acc = remove_gravity(common_time, acc, quaternions)

plot_imu_data(common_time, gyro, acc, mag, motion_acc, quaternions)