# Automatic classification Pipeline
##Projet3A_ML
A classification pipeline that performs automatic data preparation and processing in addition to training the best selected model via grid search and report calssification results

Made by  : 

Maachou Marouane <br/>
Bensaid Reda <br/>
Jallouli Mouad <br/>
Taoufik Moad <br/>


To use the main on both datasets : </br>

You should create a data folder where you should put the two data sets </br>

for the kidney disease dataset :</br></br>

python3 main.py "data/kidney_disease.csv"  "classification"  "True"  "0.2"  "mean"  "label"  "Mean" "0.95" "f1_score"                                    </br>

for the banknote dataset:</br></br>

python3 main.py "data/data_banknote_authentication.txt" "4" "False" "0.2" "mean" "label" "Mean" "0.95" "f1_score"</br></br>


To modify the parameters of the pipeline:</br></br>

the meaning of each value in order:</br></br>

- path to dataset</br>
- name of the columns corresponding to the annotation</br>
- if there are headers in the dataset corresponding to column names : True for kidney disease dataset , False for banknote dataset</br>
- test size</br>
- the missing data strategy used in preprocessing can be one of  ["","mean", "median","radical"]</br>
- the encoding strategy for categorical columns can be one of ["","label","onehot"]</br>
- the normalizinf strategy used in preprocessing can be one of ["","Mean", "MinMax"]</br>
- the percentage of variance retained after PCA or number of dimension retained after PCA should be between 0 and 1 or an integer</br>
- the metric used for evaluating the model </br>
