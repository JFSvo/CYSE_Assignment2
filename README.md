#### Steps to run the model ####

1. Run "git clone https://github.com/JFSvo/CYSE_Assignment2.git" in the directory where you want to download the repository. It may take a while since the model is 475MB. 
2. Check the file size of 5_epoch_fine_tune_sentiment_model.pt to confirm it is 475MB. It may be a placeholder stub if Git Large File Storage is not installed. If this is the case, download the raw model file directly from https://github.com/JFSvo/CYSE_Assignment2/raw/refs/heads/main/model_checkpoint/5_epoch_fine_tune_sentiment_model.pt?download= and place it in the model_checkpoint folder.
3. Navigate to the CYSE_Assignment2 directory.
4. Create a python virtual environment using "python -m venv .venv/"
5. Activate the virtual environment.
6. Run "pip install -r requirements.txt".
7. Run all modules in the stage1_notebook.ipynb Jupyter notebook with that virtual environment selected.
