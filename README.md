CONFIGURABLE AUTOMATED FEATURE (CAFR) FRAMEWORK 

Graph Neural Network to recognize machining features in .stl CAD data
1. Create a folder called data with the following structure the CAD-, Label-, and Graph data:
      1. data
         1. cad
            1. training
            2. test
         2. graph
            1. training
               1. test
   This CAFR framework follows the CAD and label data naming definition defined by https://github.com/PeizhiShi/MsvNet 
   If you want to use FeatureNet dataset from https://github.com/zibozzb/FeatureNet you need to add .csv files for each
   .stl file in the FeatureNet dataset. Also, you have to change the .STL file ending for each CAD model to .stl
   
   For easier implementation, we prepared the FeatureNet dataset with the right naming and fitting label files as well 
   as training and test data created by this framework via https://github.com/Boehm92/cafr_framework_datasets

2. Create a python virtual environment for the synthetic_data_generator:
   1. Environment should be created with an interpreter for python 3.10
   2. Activate the environment and install the requirements.txt file from the synthetic_data_generator application.
      To do so, you must change directories to be in the same directory as the requirements.txt file and type into the
      command line pip install -r requirements.txt
   3. You also can install the packages by hand:
      1. pip install pymadcad==0.16.0 
      2. pip install numpy-stl==3.0.1 
      3. pip install pandas==2.1.1

3. Create a python virtual environment for the graph_neural_network:
   1. Environment should be created with an interpreter for python 3.10
   2. Activate the environment and install the requirements.txt file from the graph_neural_network application.
      Important: Because we need for this application, two different pip wheels, one for torch and one for 
      pytorch geometric, we provide to different files: requirements1.txt and requirements2.txt 
      To install the necessary python packages you must change directories to be in the same directory as the 
      requirements.txt files and type into the command line pip install -r requirements1.txt and after that
      pip install -r requirements.txt . IMPORTANT HINT: We weren't able to install the torch package via the 
      requirements1.txt file and had to install ist manually, due to the fact that every time the torch cpu version was 
      installed. If you want to train on your gpu use following command with fitting wheel
      pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html . You find more 
      information for installing this package at https://pytorch.org/get-started/previous-versions/ . NOTE: The cuda
      version is specific to your graphics card and what version you have installed. So please change the wheel link
      accordingly
   3. You also can install the packages by hand:
      1. pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html 
      2. pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric 
      -f https://data.pyg.org/whl/torch-1.12.1%2Bcu113.html
      3. pip install wandb==0.13.9 (wandb sends the training data to the wandb-webserver, so you have to log in via 
         the terminal with following command: wandb login, also you have to register at wandb)
      4. pip install optuna==3.1.0
      5. pip install numpy-stl==3.0.0
   4. IMPORTANT: The packages here can differ to do graphic card settings and cuda version. However, it should always
      be possible to run the code on the CPU. For more information, please visit: 
      https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

4. Define environment variables for the in steps 2 and 3 created virtual environments and add the directories for the 
   in step 1 created directories. The framework needs in total five environment variables:
   1. TRAINING_DATASET_SOURCE = data -> cad -> training (both data generator and graph neural network application)
   2. TEST_DATASET_SOURCE = data -> cad -> test  (both data generator and graph neural network application)
   3. TRAINING_DATASET_DESTINATION = data -> graph -> test  (only graph neural network application)
   4. TEST_DATASET_DESTINATION = data -> graph -> test (only graph neural network application)
   5. WEIGHTS = graph_neural_network -> scripts -> network_model (only graph neural network application)

5. Now the applications are set up and ready to run. Either use the provided data by Zangh et al. (FeatureNet) 
   https://github.com/madlabub/Machining-feature-dataset/tree/master or use your own CAD data. 
   In either case you have to put the data in data -> cad -> training and test. You also can use the data generator 
   application to create new data. With the defined environment variables the data should bea automatically be put into 
   the right directories. To create the necessary training and test data change into the "synthetic_data_generator" and
   runt the __main__.py file

6. To train the graph neural network make sure the folders from data -> graph -> training and test are empty. Then run
   the __main__.py file in the graph_neural_network folder. This should start automatically the conversion of the cad
   files into fitting graph data. The graph data is then saved in the graph -> training and test directories and will be
   used for every following run. Following the data conversion the training should automatically start. If you 
   registered at https://wandb.ai/site and logged in via the command line you can follow the training process via a link
   which should be printed in the console. For more information about the wandb registration and logging please visit 
   https://docs.wandb.ai/quickstart

7. We also provide docker files to run the applications in a docker container. However, this is barely tested, and it 
   can't guarantee that it work on your system as it this at ours