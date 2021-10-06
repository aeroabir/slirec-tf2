## Adaptive User Modeling with Long and Short-Term Preference for Personalized Recommendation
This code provides an implementation of the SLi-Rec network for sequential recommendation in [this paper](https://www.microsoft.com/en-us/research/uploads/prod/2019/07/IJCAI19-ready_v1.pdf). <br />
This is a Tf 2.3.0 version of the original codebase [TF 1.4.1](https://github.com/zepingyu0512/sli_rec) <br />
The original version is also available in: <br />
https://github.com/microsoft/recommenders/tree/master/reco_utils/recommender/deeprec/models/sequential .

## Data Preparation
```
sh data_preparing.sh
```
In data/, it will generate these files: 
- reviews_information
- meta_information
- train_data 
- test_data
- user_vocab.pkl 
- item_vocab.pkl 
- category_vocab.pkl 

Note: The original training data has all the sub-sequences as training examples. If you want to retain only the
last example you can use reduce_data.py which will create a new training data. The test data already contains the
last example and no need to shorten it further. 

## Model Implementation
The model is implemented in ```model_tf2.py```, which uses a custom LSTM model with additional weights to process 
the time information (```talstm.py```).
Training model:
```
python sli_rec/train_tf2.py
```

After training, run the following code to evaluate the model:
```
python sli_rec/test.py
```

The model below had been supported: 

Baselines:
- ASVD
- DIN
- LSTM
- LSTMPP
- NARM
- CARNN
- Time1LSTM
- Time2LSTM
- Time3LSTM
- DIEN

Our models:
- A2SVD
- T_SeqRec
- TC_SeqRec_I
- TC_SeqRec_G
- TC_SeqRec
- SLi_Rec_Fixed
- SLi_Rec_Adaptive (currently supported)

## Dependencies (other versions may also work):
- python==3.7.7
- tensorflow==2.4.0
- keras==2.1.5
- numpy==1.18.5
