# AckermanLab

Test code using `python main.py classifier permuting synthetic_subject_count` where classifier is 1-6 based on the classifier you wish to use (see parse_csv.py for details), permuting is 0 for not permuting and 1 for permuting, and synthetic_subject_count is the integer of subjects you wish to add.

Important files:

* main.py contains the background code needed to make most everything run
* ml_functions.py contains a number of machine learning functions as well as some helper functions like get_holdouts and add_artificial_subjects
* parse_csv.py creates useful dataframes and output files, establishing attributes, classifiers, means, and standard deviations
* RNN.py contains the means for running neural networks and hopefully an actual RNN soon
* evaluate.py allows for collecting the accuracy and loss for hundreds of runs of a model
