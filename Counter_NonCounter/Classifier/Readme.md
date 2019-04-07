Contains all the files related to the task of classifying if a given youtube comment in response to a hate speech video is a counterspeech or non-counterspeech.


```
./vocab_lstm.pkl                        --> contains the vocabulary of the lstm model
./Counterspeech_Binary.ipynb            --> contains the classification performance of the binary classification using non-deep learning models
./Counterspeech_Binary_LSTM.ipynb       --> contains the classification performance of the binary classification using deep learning model-LSTM
./Counterspeech_Crosscommunity.ipynb    --> contains the classification performance of the binary classification of cross community predictions 
./all_preds_lstm.pkl                    --> contains all the predicitons of the 10-fold cross validation used for direct result generation of lstm model
./all_preds_binary.pkl                  --> contains all the predicitons of the 10-fold cross validation used for direct result generation of the best binary model(refer paper)
./lgbt_all_save.pkl                     --> contains all the predicitons of the lgbt category where the model was trained on jew and black category.
./black_all_save.pkl                    --> contains all the predicitons of the black category where the model was trained on jew and lgbt category
./jew_all_save.pkl                      --> contains all the predicitons of the jew category where the model was trained on black and lgbt category
```
