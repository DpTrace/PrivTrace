import numpy as np
from data_preparation.trajectory_preparation import TrajectoryPreparation
from config.parameter_setter import ParSetter

if __name__ == "__main__":
    args = ParSetter().set_up_args(epsilon=1)
    trapre1 = TrajectoryPreparation()
    trapre1.get_trajectory(args)
