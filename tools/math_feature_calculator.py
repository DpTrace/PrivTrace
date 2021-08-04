import numpy as np
import config.folder_and_file_names as config
from tools.object_store import ObjectStore
from tools.general_tools import GeneralTools


class MathFeatureCalculator:

    #
    def __init__(self):
        self.gt = GeneralTools()

    # this function calculates step lengths of gps trajectory
    def calculate_step_lengths(self, gps_array: np.ndarray):
        trajectory_length = gps_array.shape[0]
        last_locations = gps_array[0: trajectory_length - 1, :]
        this_location = gps_array[1: trajectory_length, :]
        displacement = this_location - last_locations
        latitude_displacement = displacement[:, 0]
        longitude_displacement = displacement[:, 1]
        latitude_displacement.reshape(-1)
        longitude_displacement.reshape(-1)
        step_length = np.sqrt(latitude_displacement ** 2 + longitude_displacement ** 2)
        return step_length