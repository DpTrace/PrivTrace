import config.folder_and_file_names as fname
from discretization.get_discretization import DisData
from primarkov.build_markov_model import ModelBuilder
from generator.state_trajectory_generation import StateGeneration
from generator.to_real_translator import RealLocationTranslator
# from tools.object_store import ObjectStore
from config.parameter_carrier import ParameterCarrier
from config.parameter_setter import ParSetter
from tools.data_writer import DataWriter
from data_preparation.data_preparer import DataPreparer
import datetime

if __name__ == "__main__":
    writer = DataWriter()
    print('begin all')
    print(datetime.datetime.now())
    par = ParSetter().set_up_args()
    pc = ParameterCarrier(par)
    data_preparer = DataPreparer(par)
    trajectory_set = data_preparer.get_trajectory_set()
    disdata1 = DisData(pc)
    grid = disdata1.get_discrete_data(trajectory_set)
    mb1 = ModelBuilder(pc)
    mo1 = mb1.build_model(grid, trajectory_set)
    mb1 = ModelBuilder(pc)
    mo1 = mb1.filter_model(trajectory_set, grid, mo1)
    sg1 = StateGeneration(pc)
    st_tra_list = sg1.generate_tra(mo1)
    rlt1 = RealLocationTranslator(pc)
    real_tra_list = rlt1.translate_trajectories(grid, st_tra_list)
    writer.save_trajectory_data_in_list_to_file(real_tra_list, fname.result_file_name)
    print('end all')
    print(datetime.datetime.now())
    pass