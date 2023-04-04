# MachiningFeatureRecognizer
Graph Neural Network to recognize machining features in .stl CAD data


data_generator
-python 3.8.10
-define environment variables with the names TEST_DATASET_SOURCE|TEST_DATASET_SOURCE with the absolute directories of
the training and test data. 
Example: TEST_DATASET_SOURCE: /home/stefan/Schreibtisch/Repos/gnn_machining_feature_extraction/data/cad_data/test
-install requirements:
    -pip install pymadcad==0.15.1
    -pip install numpy-stl==3.0.0
    -pip install pandas==1.5.3

graph_neural_network
-python 3.8.10
-define environment variables with the names TEST_DATASET_SOURCE|TEST_DATASET_SOURCE|TRAINING_DATASET_DESTINATION
|TESTDATASET_DESTINATION with the absolute directories of the training and test data. 
Example: TEST_DATASET_SOURCE: /home/stefan/Schreibtisch/Repos/gnn_machining_feature_extraction/data/cad_data/test

-install requirements for cuda on linux:
    - pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
    - pip install torch==1.13 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
    - pip install wandb==0.13.9 (wandb sends the training data to the wandb-webserver, so you have to log in via the 
the terminal with following command: wandb login, also you have to registere at wandb)
    - pip install optuna==3.1.0
    - pip install numpy-stl==3.0.0

Before running the program, please execute the following installation guide
1. Create a virtual environment based on python 3.10
2. Select the main.py file as script path
3. type pip install -r requirements.txt into your terminal while your virtual environment is selected
4. Depending on your operation system, if you use pip or Conda, your preferred PyTorch version and if you use 
your CPU or a specific CUDA version you have to change the wheel links in the requirements file.
Therefor, just select your 	preferred configuration at https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
and copy ONLY the shown "-f -f https://data.pyg.org/.....". Exchange the copied link with the first
link in the requirements file

