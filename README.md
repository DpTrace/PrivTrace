# PrivTrace

This is the project code for PrivTrace: Differentially Private Trajectory Synthesis by Adaptive Markov Model

Instructions:

1.install and use anaconda

* create environment in the server: conda env create -f environment.yml; source activate db_code

* share the environment: conda env export > environment.yml

* or create a new one: conda create -n myenv; conda activate myenv; conda install numpy, pandas, scipy, matplotlib


2.how to run the code

* entry is main.py

* we provide a small example dataset to run the program

* the input dataset should be put in ./datasets

* for the format of trajectory data, refer to ./datasets/data_format.txt

* the name of the input and output file can be changed in ./config/folder_and_file_names.py

* the name of the input file can also be a parameter of main.py, like "python main.py --dataset_file_name=simple_example.dat"

* by default, the output file will appear in this folder with the name "generated_tras.txt"
