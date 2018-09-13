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

# 4. How to run
You can use "pyton3 LEARN_XXX --num YYY" to run a model.