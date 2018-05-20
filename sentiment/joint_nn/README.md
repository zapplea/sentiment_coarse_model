#1. Optimizer
graph.get_collection('senti_opt') can get optimizer for sentiment loss
graph.get_collection('opt') can get optimizer for attribute loss
graph.get_collection('joint_opt') can get optimizer fo joint loss

#2. Parameters:
###2.1 new parameters:
'joint_lr': learning rate for joint loss
'lr': learning rate for attribute loss
'senti_lr': learning rate for sentiment loss

###2.2 basic parameters:
combination of paramters in attribute function and sentiment function

#3. Training
1. train attribute loss
2. train joint loss
