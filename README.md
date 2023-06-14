# HRTEM-Generalization

This repo hosts various codes and notebooks supporting our manuscript, "Generalization Across Experimental Parameters in Machine Learning Analysis of High Resolution Transmission Electron Microscopy Datasets". 

**dataset_creation** is a folder that contains 1) code to download the raw HRTEM images of nanoparticles from our data repository on NERSC, and 2) our preprocessing steps to convert HRTEM images into a dataset ready for machine learning. 

**train_test_model_example.ipynb** is a Jupyter notebook that explains how we train and test our UNet models, showing as an example how we trained on the 2.2nm nanoparticle dataset and tested on datasets of other nanoparticle sizes. 

**trained_models** is a folder that contains all of our trained model weights, with **model_details.csv** explaining which dataset and training/valid/test split trained each model. 

**noise_augmentation_notebook.ipynb** is a Jupyter notebook that shows how we implemented addtive Gaussian noise augmentation to our training procedure, our model generalization results, and dataset analysis. 

**hyperparameter_search_notebook.ipynb** is a Jupyter notebook that shows our hyperparameter tuning results (from the SI of the manuscript). 
