# How to Extract the Splits

All data MCHR1-based data for this project is stored in `20240430_MCHR1_splitted_RJ.csv`. That file contains columns called `random_10f` and `DataSAIL_10f`. Those contain the splitting for 10-fold CV computed using a random splitter and DataSAIL, respectively. The latter is based on similarities between the ligands and clustered the ligands before assigning the clusters to splits. Therefore, it's an information leakage-reduced split.

## How to get the i-th fold from the splitted data?

The `XXX_10f` column contains values from `Fold_0` to `Fold_9`. Each fold is constructed in a 7:1:1:1 manner, where 

  * 70% go into model training, 
  * 10% go into model validation,
  * 10% are reserved for the ensembling (always `Fold_8`) and the last 
  * 10% are used for testing the models and the whole ensemble (always `Fold_9`).

In the `i`-th fold, the labels

  * `Fold_i` to `Fold_i+6` are the training split,
  * `Fold_i+7` is the validation split
  * `Fold_8` is the ensembling split, and
  * `Fold_9` is the test split.

All `i`'s must be taken modulo 7. Therefore, in the 5-th fold the assignment of `Fold_X` looks like this

  * Training: `Fold_5`, `Fold_6`, `Fold_7`, `Fold_0`, `Fold_1`, `Fold_2`, `Fold_3`
  * Validation: `Fold_4`
  * Ensembling: `Fold_8`
  * Testing: `Fold_9`

DO NOT USE `Fold_8` or `Fold_9` (!) for your model development. That constitutes Information Leakage since the model and hyperparameter selection is based on the test data that is later used to assess the quality of the hyperparameters.

