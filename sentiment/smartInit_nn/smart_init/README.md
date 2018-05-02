## What is new?
In the past design, the attributes mention matrix is initialized randomly.

## How does the smartInit work?
for each aspect#attribute, we generate a name list. In name list we choose k words that is related to aspect and attribute.
Then for each sentence, we give it a name_list_vector, the shape of name list vector is: (attributes number,). Its each scalar 
represent existence of a "aspect#attribute". If the aspect#attribute exists, then corresponding value is 1 otherwise is 0.
We use words in the sentence to determine whether the "aspect#attribute" exists. If word_i in the sentence can be found in name list of 
aspect#attribute_j, them name_list_vec[j]=1.
ID of "aspect#attribute" in name_list_vec is the same to output layer.


TODO:
1. need to use pmi_idf to find a name list for  each aspect#attribute. for each aspect#attribute, they can choose words from top k words in pmi_idf list,
and put them in the name list. For different aspect#attribute, the k doesn't need to be the same.
2. for each sentence, check each word in it to create name list vector.
3. after create the name list vector for each sentence, input it to name_list_vec in sess.run.
