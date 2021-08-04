import numpy as np
from tools.data_reader import DataReader
from tools.data_writer import DataWriter
from primarkov.mar_model import MarkovModel
from primarkov.guidepost import GuidePost
from generator.trajectory_generator import Generator
import config.folder_and_file_names as config
from config.parameter_carrier import ParameterCarrier
from tools.object_store import ObjectStore


class StateGeneration:

    #
    def __init__(self, cc: ParameterCarrier):
        self.cc = cc

    def generate_tra(self, mar_mo):
        cc1 = self.cc
        # reader = DataReader()
        # mar_mo_file_name = config.filtered_model_name + '.txt'
        # mar_mo = reader.read_files_from_path(mar_mo_file_name)

        # os1 = ObjectStore()
        # mar_mo = os1.load_markov_model()
        generator1 = Generator(self.cc)
        generator1.load_generator(mar_mo)

        # tr1 = generator1.generate_trajectory(neighbor_check=True)
        number = cc1.trajectory_number_to_generate
        usable_tr_list = generator1.generate_many(number, neighbor_check=False)
        print('state trajectories got')
        real_tr_list = self.trans_many_usable_trajectories(usable_tr_list, mar_mo.grid)
        # data_writer = DataWriter()
        # data_writer.save_to_file(real_tr_list, config.state_trajectories_name)

        # os1.save_state_trajectories(real_tr_list)
        return real_tr_list

    # this function transfers a usable state trajectory to real state trajectory
    def trans_to_real_state_trajectory(self, usable_to_real_dict: np.ndarray, usable_state_trajectory: np.ndarray):
        real_state_trajectory = usable_to_real_dict[usable_state_trajectory]
        return real_state_trajectory

    # this function transfers a list of usable state trajectory to real state trajectory
    def trans_many_usable_trajectories(self, usable_state_trajectories: list, grid1) -> list:
        # os1 = ObjectStore()
        # grid1 = os1.load_grid()
        usable_to_real_dict = grid1.usable_subcell_index_to_real_index_dict
        tr_list2 = []
        for usable_trajectory in usable_state_trajectories:
            real_trajectory = self.trans_to_real_state_trajectory(usable_to_real_dict, usable_trajectory)
            tr_list2.append(real_trajectory)
        return tr_list2


# if __name__ == "__main__":
#     reader = DataReader()
#     mar_mo_file_name = 'filtered_markov_model' + '.txt'
#     mar_mo = reader.read_files_from_path(mar_mo_file_name)
#     generator1 = Generator()
#     generator1.load_generator(mar_mo)
#     # tr1 = generator1.generate_trajectory(neighbor_check=True)
#     number = 4000
#     tr_list1 = generator1.generate_many(number, neighbor_check=False)
#     ln1 = np.zeros(6)
#     for tr in tr_list1:
#         for l in range(6):
#             if tr.size > l:
#                 ln1[l] = ln1[l] + 1
#     ratio = ln1 / number
