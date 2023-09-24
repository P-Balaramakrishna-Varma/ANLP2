## Folder structure
- Dataset
    - AGTokenizedData: The AG dataset is preprocessed for downstream task and is stored as json in this directory.
    - Code: Contains code for preporcessing the AG dataset. It creates the files in AGTokenizedData and LMTokenizedData.
    - Dataset: Contains the AG dataset in csv format.
    - LMTokenizedData: The AG dataset is preprocessed for language modeling task in and stored as json in this directory.
- Downstream
    - classification.py: This file contains the code for the downstream task. This contains code for data processing, model and tranning.
- ELMov
    - backward.py: This file is used to train the backward LM.
    - elmov.py: This file contains the ELMov architecture. Loads the forward and backward LM.
    - forward.py: This file is used to train the forward LM.
    - LM.py: This file contains code for data processing, LM models. This is also used in forward.py and backward.py.
- Plots
    - Backward: This directory contains plots for accuracy, perplexity, loss for backward LM for 1 experiment.
    - DownStream: This directory contains plots for accuracy, loss for downstream task for 2 experiments.
    - Forward: This directory contains plots for accuracy, perplexity, loss for backward LM for 5 experiments.
- theory.md: Answers to theory question section 3.1
- environment.yml: conda environment file
- Readme.md: Code base structure.


## Instructions to run
- All the python files run without any arguments.
- Change parameters and file paths in the python files to run the code for experimentation.

## OneDrive link for the models
- https://iiitaphyd-my.sharepoint.com/:f:/g/personal/balaramakrishna_p_students_iiit_ac_in/EgsLFnW4MqNNmRWnxB7qVgIBGIF-Lfn65M6J-Dxo4frVvA?e=VQ0J09