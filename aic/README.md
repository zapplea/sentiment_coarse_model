# 1. Strucuture of one complete neural network model
One model can be splitted to five pieces: metrics, net, train, learn, and data_process.

The metrics module include ways of evaluation.
The net include the neural network model;
The train include training process;
The data_process include method to provide data;
The learn  composes all parts together to form a applicable neural network model.

Note:Tensorflow is used only in net and train module.

# 2. Parameters
The config is splited into three parts and stored in net, train, and data process. You can specify a parameter in learn and the 
program will update the corresponding parameters in net/train/data process.

# 3. About data
The data should be converted from text to id format and then fed to the model. In this program, we assume that the train, test, and validation 
data has been generated and only is the datafeeder module implemented to feed data. The data preprocess programs can be included under directory "data_preprocess"

# TODO:
1. for sentiment training: make sure senti_loss,joint_loss, attr_loss was added with comm_reg. 
the last element in graph.get_collection('reg')
2. recheck the sentiment labels input. data and the net.
should I add non-attribute sentiment label?
current solution to non-attribute sentiment: when train, the none-attribute sentiment labels will be 0 which is similar to a zero mask, so when the non-attribute
is detected, there will not be update to the sentiment model it self. Check the none-attribute sentiment matrix. 

the case realted to the none-attribute:
There is not attribute in the Review, but the review is opinioned. So, the input size should be ()attribute num + 1,)
should not mask sentiment of none-attribute

what's the behavior of the same variable when multiple gpu is used.

3. add dropout in the bilstm

4. for elmo
confirm bidirectional in hyperparameter

# Question:
1. we use sentiment_bilstm twice for sentiment and attribute seperately, should I combine them to one?