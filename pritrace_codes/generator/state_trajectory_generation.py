import numpy as np
from generator.trajectory_generator import Generator
from config.parameter_carrier import ParameterCarrier


class StateGeneration:

    def __init__(self, cc: ParameterCarrier):
        self.cc = cc

    def generate_tra(self, mar_mo):
        cc1 = self.cc
        generator1 = Generator(self.cc)
        generator1.load_generator(mar_mo)
        number = cc1.trajectory_number_to_generate
        usable_tr_list = generator1.generate_many(number, neighbor_check=False)
        print('state trajectories got')
        real_tr_list = self.trans_many_usable_trajectories(usable_tr_list, mar_mo.grid)
        return real_tr_list

    # this function transfers a usable state trajectory to real state trajectory
    def trans_to_real_state_trajectory(self, usable_to_real_dict: np.ndarray, usable_state_trajectory: np.ndarray):
        real_state_trajectory = usable_to_real_dict[usable_state_trajectory]
        return real_state_trajectory

    # this function transfers a list of usable state trajectory to real state trajectory
    def trans_many_usable_trajectories(self, usable_state_trajectories: list, grid1) -> list:
        usable_to_real_dict = grid1.usable_subcell_index_to_real_index_dict
        tr_list2 = []
        for usable_trajectory in usable_state_trajectories:
            real_trajectory = self.trans_to_real_state_trajectory(usable_to_real_dict, usable_trajectory)
            tr_list2.append(real_trajectory)
        return tr_list2