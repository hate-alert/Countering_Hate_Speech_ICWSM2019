[![HitCount](http://hits.dwyl.io/binny-mathew/Countering_Hate_Speech.svg)](http://hits.dwyl.io/binny-mathew/Countering_Hate_Speech)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/thou-shalt-not-hate-countering-online-hate-1/counterspeech-detection-on-youtube)](https://paperswithcode.com/sota/counterspeech-detection-on-youtube?p=thou-shalt-not-hate-countering-online-hate-1)
# Thou shalt not hate: Countering Online Hate Speech

Binny Mathew, Punyajoy Saha, Hardik Tharad, Subham Rajgaria, Prajwal Singhania, Suman Kalyan Maity, Pawan Goyal, and Animesh Mukherjee. 2019. "[Thou shalt not hate: Countering Online Hate Speech](https://arxiv.org/abs/1808.04409)". ICWSM

***Please cite our paper in any published work that uses any of these resources.***

> Mathew, B., Saha, P., Tharad, H., Rajgaria, S., Singhania, P., Maity, S. K., Goyal, P., & Mukherje, A. (2019). Thou shalt not hate: Countering online hate speech. In _Thirteenth International AAAI Conference on Web and Social Media (ICWSM)_.

~~~bibtex
@inproceedings{mathew2018thou,
  title={Thou shalt not hate: Countering online hate speech},
  author={Mathew, Binny and Saha, Punyajoy and Tharad, Hardik and Rajgaria, Subham and Singhania, Prajwal and Maity, Suman Kalyan and Goyal, Pawan and Mukherje, Animesh},
  booktitle={Thirteenth International AAAI Conference on Web and Social Media},
  year={2019}
}

~~~


------------------------------------------
***Folder Description***
------------------------------------------
~~~

./Dataset             --> Contains the dataset
./Counter_NonCounter  --> Contains the classifiers performing binary classifier on the total dataset
./Multilabel          --> Contains the classifiers performing the multilabel classification of counter speech 
./Best_model          --> Contains the best models we found for each of the three tasks
./Full_Results        --> Contains the classification results for all classifiers and other dataset related information
./Utils               --> Miscellaneous functions used for the task.

~~~


## Requirements 

Make sure to use **Python3** when running the scripts. The package requirements can be obtained by running `pip install -r requirements.txt`.


------------------------------------------
***Demo Available***
------------------------------------------

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/binny-mathew/Countering_Hate_Speech/blob/master/DEMO_Counter_speech.ipynb)

------------------------------------------
***Instructions for using the models***
------------------------------------------

1. **Binary classifier model**  
    1. Load the data into the python script or the jupyter notebook. If the data is in .json format convert the data into .csv. After converting into .csv format check the column names and column values in few rows if that matches with the **dataset** available here.
    2. Load the binary classification model using joblib as shown in **DEMO_Counter_speech.ipynb** 
    3. Load the word embeddings model by using the function **loadGloveModel2** and changing the path to where your word embeddings are present. Here the [glove](https://nlp.stanford.edu/projects/glove/)  embeddings trained on 840 billion sentences and having dimension as 300 is used.  
    4. Import following sub-modules from utils module
       1. features - here function to extract the binary features and their combination are present
       2. tokenize - here different tokenization are used 
       3. helper - here results saving functions are used
       4. commen_preprocess - here preprocessing methods for text are used 
    5. from features module import **combine_tf_rem_google_rem_embed** and use it to generate the feature matrix **X** and decision vector **y** as in **DEMO_Counter_speech.ipynb**.
    6. run the model on the feature matrix using the **predict** function and evaluate the predicted vector using the functions in the **helper** submodule.

2. **Multilabel classifier model**  
    1. Load the data into the python script or the jupyter notebook. If the data is in .json format convert the data into .csv. After converting into .csv format check the column names and column values in few rows if that matches with the **dataset** available here.
    2. Load the multilabel classification model using joblib as shown in **DEMO_Counter_speech.ipynb** 
    3. Load the word embeddings model by using the function **loadGloveModel2** and changing the path to where your word embeddings are present. Here the [glove](https://nlp.stanford.edu/projects/glove/)  embeddings trained on 840 billion sentences and having dimension as 300 is used.  
    4. Import following sub-modules from utils module
       1. multi_features - here function to extract the multilabel features and their combination are present
       2. tokenize - here different tokenization are used 
       3. helper - here results saving functions are used
       4. commen_preprocess - here preprocessing methods for text are used 
    5. Remove all the rows having **Default** category as they are not having counter speech. Rest rows should have one or more labels seperated by comma(,). 
    6. Each multilabel value is converted to a horizontal vector whose each element represent whether that numbered label is present or not. *For example*, a multilabel (2,3,9) will be converted to a vector [0,1,1,0,0,0,0,0,1,0]. This has to be done for each row in the data. It is already present in **DEMO_Counter_speech.ipynb**.
    7. from multi_features module import **combine_tf_rem_google_rem_embed** and use it to generate the feature matrix **X** and decision matrix **y** as in **DEMO_Counter_speech.ipynb**.
    8. run the model on the feature matrix using the **predict** function and evaluate the predicted vector using the functions in the **helper** submodule.
