[![HitCount](http://hits.dwyl.io/binny-mathew/Countering_Hate_Speech.svg)](http://hits.dwyl.io/binny-mathew/Countering_Hate_Speech)

# Thou shalt not hate: Countering Online Hate Speech

Binny Mathew, Punyajoy Saha, Hardik Tharad, Subham Rajgaria, Prajwal Singhania, Suman Kalyan Maity, Pawan Goyal, and Animesh Mukherjee. 2019. "[Thou shalt not hate: Countering Online Hate Speech](https://arxiv.org/abs/1808.04409)". ICWSM

***Please cite our paper in any published work that uses any of these resources.***
~~~
@article{mathew2018thou,
  title={Thou shalt not hate: Countering online hate speech},
  author={Mathew, Binny and Saha, Punyajoy and Tharad, Hardik and Rajgaria, Subham and Singhania, Prajwal and Maity, Suman Kalyan and Goyal, Pawan and Mukherje, Animesh},
  journal={arXiv preprint arXiv:1808.04409},
  year={2018}
}

~~~



------------------------------------------
***Folder Description***
------------------------------------------
~~~
/Counter_NonCounter --> contains the classifiers performing binary classifier on the total dataset
/Multilabel    --> Contains the classifiers performing the multilabel classification of counter speech 
~~~
------------------------------------------
***File Description***
------------------------------------------
~~~
./Counterspeech Video Statistics.csv  --> Contains details such as video title, URL, Number of likes etc. for the videos selected for Annotation.
./Counter_NonCounter/Classifier/vocab_lstm.pkl --> contains the vocabulary of the lstm model
./Counter_NonCounter/Classifier/commen_preprocess.py --> contains the preprocessing of tweets 
./Multilabel/commen_preprocess.py --> contains the preprocessing of tweets 
./Counter_NonCounter/Classifier/tfidf_char_vectorizer.pk --> tfidf vectorizer having 500 char features
./Counter_NonCounter/Classifier/tfidf_word_vectorizer.pk --> tfidf vectorizer having 500 word features
./Counter_NonCounter/tfidf_char_vectorizer.pk --> tfidf vectorizer having 10000 char features
./Counter_NonCounter/tfidf_word_vectorizer.pk --> tfidf vectorizer having 10000 word features
./Counter_NonCounter/Data/Counterspeech_Dataset.json --> the total dataset in json format
~~~
------------------------------------------
***Jupyter Notebooks***
------------------------------------------

~~~
./Counter_NonCounter/Classifier/Counterspeech_Binary.ipynb --> contains the classification performance of the binary classification using non-deep learning models
./Counter_NonCounter/Classifier/Counterspeech_LSTM.ipynb --> contains the classification performance of the binary classification using deep learning model-LSTM
./Counter_NonCounter/Classifier/Counterspeech_Crosscommunity.ipynb --> contains the classification performance of the binary classification of cross community predictions 
./Multilabel/Counterspeech_Multilabel.ipynb--> contains the classification performance of the multilabel classification of counterspeech samples 
./Multilabel/Data_exploration_II.ipynb--> contains some basic data exploration done in the module
~~~

------------------------------------------
***Files for predictions***
------------------------------------------
~~~
./Counter_NonCounter/Classifier/all_preds_lstm.pkl --> contains all the predicitons of the 10-fold cross validation used for direct result generation of lstm model
./Counter_NonCounter/Classifier/all_preds_binary.pkl --> contains all the predicitons of the 10-fold cross validation used for direct result generation of the best binary model(refer paper)
./Counter_NonCounter/Classifier/lgbt_all_save.pkl --> contains all the predicitons of the lgbt category where the model was trained on jew and black category.
./Counter_NonCounter/Classifier/black_all_save.pkl --> contains all the predicitons of the black category where the model was trained on jew and lgbt category
./Counter_NonCounter/Classifier/jew_all_save.pkl --> contains all the predicitons of the jew category where the model was trained on black and lgbt category
~~~





------------------------------------------
***Counterspeech Dataset***
------------------------------------------

Our dataset is available at ./Data/Counterspeech_Dataset.json. The file contains youtube comments that are labelled as either 'counterspeech' or 'non-counterspeech'. Each comment is in the form of key-value pair. The details of each such comment are as follows:
~~~
Community             -->  The community which was targetted in the YouTube Video
CounterSpeech         -->  This field indicates if the comment is a counterspeech or not.
Category              -->  The type of counterspeech used in the comment text. The possible values are : 1 - Presentation of fact, 2 - Pointing out hypocrisy or contradiction, 3 - Warning of offline or online consequences, 4 - Affiliation, 5 - Denouncing hateful or dangerous speech, 7 - Humor, 8 - Positive Tone, 9 - Hostile Language, Default - Non-Counterspeech.
commentText           -->  The comment tagged by the annotator.
hasReplies            -->  A boolean field indicating if the comments has received any replies or not.
id                    -->  The unique id assigned to each comment.
likes                 -->  The number of likes received by the comment.
numberOfReplies       -->  The number of replies received by the comment.
replies               -->  The list of replies received by the comment.
timestamp             -->  The timestamp of the comment
user --> The account name of the user who posted the comment.
~~~




------------------------------------------
***Demo***
------------------------------------------

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/binny-mathew/Countering_Hate_Speech/blob/master/RESULTS.ipynb)
