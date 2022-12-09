import numpy as np
from tools.data_reader import DataReader
from tools.data_writer import DataWriter
import config.folder_and_file_names as config
from generator.trajectory_generator import Generator
from generator.state_trajectory_generation import StateGeneration
from discretization.grid import Grid
# from tools.object_store import ObjectStore
from tools.general_tools import GeneralTools


class RealLocationTranslator:

    def __init__(self, cc):
        self.grid = Grid(cc)

    # this function gives translator grid to use
    def load_translator(self, grid):
        self.grid = grid

    def translate_given_state_sequence(self, state_sequence):
        grid1 = self.grid
        if state_sequence.size < 2:
            start_end_array = np.empty(2, dtype=int)
            start_end_array[0] = state_sequence[0]
            start_end_array[1] = state_sequence[0]
            state_sequence = start_end_array
        all_level2_state_borders = grid1.level2_borders
        trajectory_length = state_sequence.size
        real_trajectory = np.random.random((trajectory_length, 2))
        for index_of_states in range(trajectory_length):
            state = state_sequence[index_of_states]
            borders = all_level2_state_borders[state]
            location = self.sample_from_a_subcell(borders)
            real_trajectory[index_of_states, :] = location
        return real_trajectory

    def sample_from_a_subcell(self, borders):
        gt1 = GeneralTools()
        north = borders[0]
        south = borders[1]
        west = borders[2]
        east = borders[3]
        x_value = gt1.sample_from_interval(west, east)
        y_value = gt1.sample_from_interval(south, north)
        location = np.array([x_value, y_value])
        return location

    def sample_with_direction(self, last_step, this_step, next_step, borders):
        gt1 = GeneralTools()
        grid1 = self.grid

        if last_step == 'start' and (not next_step == 'end'):
            direction = grid1.subcell_direction(this_step, next_step)
        elif next_step == 'end' and (not last_step == 'start'):
            direction = grid1.subcell_direction(this_step, last_step)
        elif (not last_step == 'start') and (not next_step == 'end'):
            last_direction = grid1.subcell_direction(last_step, this_step)
            next_direction = grid1.subcell_direction(this_step, next_step)
            if (last_direction == 'w' and next_direction == 'e') or (last_direction == 'e' and next_direction == 'w'):
                direction = 'ew'
            elif (last_direction == 's' and next_direction == 'n') or (last_direction == 'n' and next_direction == 's'):
                direction = 'ns'
            else:
                direction = next_direction
        else:
            direction = 'no'
        north, south, west, east = self.get_biased_borders(borders, direction)
        x_value = gt1.sample_from_interval(west, east)
        y_value = gt1.sample_from_interval(south, north)
        location = np.array([x_value, y_value])
        return location

    def sample_centrally(self, borders):
        north = borders[0]
        south = borders[1]
        west = borders[2]
        east = borders[3]
        north1 = south + 2/3 * (north - south)
        south1 = south + 1/3 * (north - south)
        west1 = west + 2/3 * (east - west)
        east1 = west + 1/3 * (east - west)
        new_borders = np.array([north1, south1, west1, east1])
        location = self.sample_from_a_subcell(new_borders)
        return location

    def get_biased_borders(self, borders, direction):
        subcell_north = borders[0]
        subcell_south = borders[1]
        subcell_west = borders[2]
        subcell_east = borders[3]
        x_distance = subcell_east - subcell_west
        y_distance = subcell_north - subcell_south
        if direction == 'n':
            north = subcell_north
            south = subcell_south + 0.5 * y_distance
            west = subcell_west
            east = subcell_east
        elif direction == 's':
            north = subcell_north - 0.5 * y_distance
            south = subcell_south
            west = subcell_west
            east = subcell_east
        elif direction == 'w':
            north = subcell_north
            south = subcell_south
            west = subcell_west
            east = subcell_east - 0.5 * x_distance
        elif direction == 'e':
            north = subcell_north
            south = subcell_south
            west = subcell_west + 0.5 * x_distance
            east = subcell_east
        elif direction == 'ns':
            north = subcell_north
            south = subcell_south
            west = subcell_west + 0.25 * x_distance
            east = subcell_east - 0.25 * x_distance
        elif direction == 'ew':
            north = subcell_north - 0.25 * y_distance
            south = subcell_south + 0.25 * y_distance
            west = subcell_west
            east = subcell_east
        else:
            north = subcell_north
            south = subcell_south
            west = subcell_west
            east = subcell_east
        return north, south, west, east

    def centralized_biased_borders(self, borders, direction):
        subcell_north = borders[0]
        subcell_south = borders[1]
        subcell_west = borders[2]
        subcell_east = borders[3]
        central_point_x = (subcell_west + subcell_east) / 2
        central_point_y = (subcell_south + subcell_north) / 2
        x_distance = subcell_east - subcell_west
        y_distance = subcell_north - subcell_south
        if direction == 'n':
            north = central_point_y + y_distance / 6
            south = central_point_y - y_distance / 6
            west = central_point_x - x_distance / 12
            east = central_point_x + x_distance / 12
        elif direction == 's':
            north = central_point_y + y_distance / 6
            south = central_point_y - y_distance / 6
            west = central_point_x - x_distance / 12
            east = central_point_x + x_distance / 12
        elif direction == 'w':
            north = central_point_y + y_distance / 12
            south = central_point_y - y_distance / 12
            west = central_point_x - x_distance / 6
            east = central_point_x + x_distance / 6
        elif direction == 'e':
            north = central_point_y + y_distance / 12
            south = central_point_y - y_distance / 12
            west = central_point_x - x_distance / 6
            east = central_point_x + x_distance / 6
        elif direction == 'ns':
            north = central_point_y + y_distance / 12
            south = central_point_y - y_distance / 12
            west = central_point_x - x_distance / 6
            east = central_point_x + x_distance / 6
        elif direction == 'ew':
            north = central_point_y + y_distance / 6
            south = central_point_y - y_distance / 6
            west = central_point_x - x_distance / 12
            east = central_point_x + x_distance / 12
        else:
            north = central_point_y + y_distance / 12
            south = central_point_y - y_distance / 12
            west = central_point_x - x_distance / 12
            east = central_point_x + x_distance / 12
        return north, south, west, east

    def get_real_trajectories(self, state_trajectories):
        state_trajectory_list = state_trajectories
        real_trajectory_list = []
        for state_trajectory in state_trajectory_list:
            real_trajectory = self.translate_given_state_sequence(state_trajectory)
            real_trajectory_list.append(real_trajectory)
        return real_trajectory_list

    def translate_trajectories(self, grid, state_trajectories):
        self.load_translator(grid)
        real_tra = self.get_real_trajectories(state_trajectories)
        return real_tra
