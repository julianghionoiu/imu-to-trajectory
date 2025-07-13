
from lib.utils import remove_gravity, load_from_dir, frequency_from_time, plot_imu_data
from ahrs.filters import AQUA

common_time, mag, acc, gyro = load_from_dir("./devices/E0A8AD21/gyr_calibration/zrot/")

aqua = AQUA(gyr=gyro, acc=acc, mag=mag, frequency=frequency_from_time(common_time))
quaternions = aqua.Q

motion_acc = remove_gravity(common_time, acc, quaternions)

plot_imu_data(common_time, gyro, acc, mag, motion_acc, quaternions)