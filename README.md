# sentiment_coarse_model

Paper: uploaded to dropbox


TODO_list:
1. Need to complete sentiment_coarse_model/sentiment/util/coarse/atr_data_generator.py. it lacks the part of generating training data, but I provide function data_generator to feed data to training process. 
2. Need to write a python file to call classifier and train it. Still, remember to add sys.path.append as in the af_unittest.py


The model to train:
1. (a-o)*e: sentiment_coarse_model/sentiment/sep_nn/coarse_atr_classifier_1w
2. (a-o)*h: sentiment_coarse_model/sentiment/sep_nn/coarse_atr_classifier_nw
3. (a-o)*h+(a-o)*e: sentiment_coarse_model/sentiment/sep_nn/coarse_atr_classifier_1pNw

Notes:
1. Can run unittest af_unittest.py for each model. 
2. Before run the unittest, need to change PATH in: sys.path.append('/home/liu121/dlnlp')
3. For training, the hyper-parameter should be the same to these in unittest, but you can change values.
