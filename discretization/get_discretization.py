from discretization.grid import Grid
from config.parameter_carrier import ParameterCarrier


class DisData:

    def __init__(self, cc: ParameterCarrier):
        self.cc = cc

    def get_discrete_data(self, trajectory_set1):
        grid = Grid(self.cc)
        grid.get_grid(trajectory_set1)
        grid.set_up_state(trajectory_set1)
        trajectory_set1.get_simple_trajectory(grid.real_subcell_index_to_usable_index_dict)
        if self.cc.trajectory_number_to_generate < 0:
            self.cc.trajectory_number_to_generate = trajectory_set1.trajectory_number
        return grid
