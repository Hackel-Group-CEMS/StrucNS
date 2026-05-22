omegafold.sh is used to get the predicted structures of protein variants.
pdb_to_graphml_batch.py is used to convert the pdb files into graphnl files containing RIN of protein variants.
graphml_to_features_base_natch.py is used to extract features from graphml files.
training.py and ddG_predict.py and are used to training and testing of the model to get predictions for dG and ddG.
run_predictions.py and predict_slurm.sh are used for getting predictions from the 6 ensemble models.
There are six model ensembles based on different seeds of dataset filtering for training as mentioned in main article.
