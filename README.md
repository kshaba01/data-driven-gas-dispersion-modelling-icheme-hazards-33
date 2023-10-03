# data-driven-gas-dispersion-modelling-icheme-hazards-33
This repo contains the source code in support of the paper presented on this work at the Instituiton of Chemical Engineer Hazards 33 Conference. 

The following material has been made available in the project git repository (available here: 

1. Source data
The source data for the Prairie Grass project is available in the following files
- “releasedata.csv”
- “measured.csv”

The processed data used as input to the final machine learning models is available here:
- “pg_ml_data_stable.csv”

2. ML modelling Source code
- prairegrass_descriptive_analytics_stable_data.py (contains the code used for preprocessing/feature selection and feature engineering)
- Comparing ML Algorithms.py (contains the code used to compare the various algorithms 
identified)
- prairegrass_data_XRT.py (contains the python script for the XRT model. Includes code to test 
a saved version of the model)
- prairegrass_data_nn.py (contains the python script for the neural network model. Includes 
code to test a saved version of the model)
- prairegrass_django_api_test.py (contains the python based API testing script)
- conda_requirements.txt (contains a list of python packages and versions used to develop the 
machine learning models)
- tensorflow375.yml (similar requirements.txt above. This contains a list of python packages 
and versions used in the conda virtual environment where this work was done)

