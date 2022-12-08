import numpy as np
from data_preparation.trajectory import Trajectory


class TrajectorySet:

    def __init__(self):
        self.trajectory_list = []
        self.trajectory_number = 0

    # this function give new trajectory number
    def refresh_trajectory_number(self):
        self.trajectory_number = len(self.trajectory_list)

    # this function give trajectory number
    def get_trajectory_number(self):
        return self.trajectory_number

    # this function give set trajectory list
    def give_trajectory_list(self, trajectory_list1):
        if not isinstance(trajectory_list1, list):
            raise TypeError('TrajectorySet must receive list of trajectory as parameter')
        self.trajectory_list += trajectory_list1
        self.refresh_trajectory_number()

    # this function add new trajectory in trajectory list
    def add_trajectory(self, trajectory1, give_index=True):
        sample1 = Trajectory()
        if type(sample1) is not type(trajectory1):
            raise TypeError('Must add trajectory to set')
        trajectory_number_now = self.get_trajectory_number()
        if give_index:
            trajectory1.give_index(trajectory_number_now + 1)
        self.trajectory_list.append(trajectory1)
        self.refresh_trajectory_number()

    # this function give trajectory according to index
    def give_trajectory_by_index(self, index1) -> Trajectory:
        try:
            trajectory1 = self.trajectory_list[index1]
        except IndexError:
            print(index1)
            raise IndexError

        return trajectory1

    # this function generate trajectories from a list of trajectory array
    def get_trajectory_set_from_data_list(self, trajectory_array_list1):
        for i in range(len(trajectory_array_list1)):
            trajectory1 = Trajectory()
            trajectory1.give_trajectory_list(trajectory_array_list1[i])
            self.add_trajectory(trajectory1)

    # this function calculate all point number in this trajectory set
    def get_whole_point_number(self) -> int:
        point_number = 0
        trajectory_number = self.get_trajectory_number()
        for trajectory_index in range(trajectory_number):
            trajectory1 = self.give_trajectory_by_index(trajectory_index)
            this_trajectory_point_number = trajectory1.get_point_number()
            point_number = point_number + this_trajectory_point_number
        return point_number

    # this function gives discrete trajectory the sample trajectory(unrepeated cell index array and its frequency)
    def get_simple_trajectory(self, dict1: np.ndarray):
        for trajectory1 in self.trajectory_list:
            trajectory1.give_simple_trajectory(dict1)

    #
    def find_trajectories_with_given_prefix(self, prefix):
        tras = []
        states_tran = np.zeros(1000)
        prefix_length = prefix.size
        end_weights = 0
        for tr in self.trajectory_list:
            seq = tr.usable_simple_sequence
            if (seq[:prefix_length] == prefix).all():
                tras.append(seq)
                if seq.size == prefix_length:
                    end_weights += 1
                else:
                    next_state = seq[prefix_length]
                    if next_state >= states_tran.size:
                        tem = np.zeros(next_state + 1)
                        tem[:states_tran.size] = states_tran
                        states_tran = tem
                    else:
                        states_tran[next_state] += 1
        return tras, states_tran, end_weights






