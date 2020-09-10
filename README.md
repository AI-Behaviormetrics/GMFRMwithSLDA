This package includes a sample dataset and Java programs for estimating the generalized MFRM that integrates the supervised latent Dirichlet allocation. 

The dataset located in the "data/" directory consists of the following three files.    
1) rating.csv: Consists of rater scores given to essays of examinees for multiple essay tasks (1st column=examinee id, 2nd column=task id, 3rd column=rater id, 4th column=observed scores).   
2) token.csv: The bag-of-words formatted essay text information (1st column=examinee id, 2nd column=task id, 3rd column=vocabulary id, 4th column=occurrence frequency of the vocabulary in the target essay).   
3) vocab.csv: List of vocabulary words in which the row number - 1 corresponds to the vocabulary id. Note that this data is not mandatory for the parameter estimation. Also, the essays composing the dataset were written in Japanese.

To estimate the model parameters using the data, run “/ctl/main_ctl.java”. Before running the code, set the number of threads specified in the “MyUtil.startThread(tasks, 10)” line according to your environment. The parameter estimates will be output to the “output” directory. 

To use your own data, prepare data with the same structure as the sample data, and save them under the "data/" directory. Then, set the variables I, J, R, K, and V in “/ctl/main_ctl.java” file to respectively indicate the numbers of tasks, examinees, raters, rating categories, and vocabularies, in accordance with your dataset.
