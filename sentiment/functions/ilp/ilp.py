import numpy as np
import pulp
import operator
# import gurobipy

class AttributeIlp:
    def __init__(self,ilp_data):
        self.ilp_data = ilp_data
        self.target_labels_num = len(ilp_data)
        self.source_vectors_num = ilp_data[0]['attention'].shape[2]*ilp_data[0]['attention'].shape[3]

    def extract_attention(self):
        """
        extract attention of each sentence
        :return: shape = (target labels num, source vectors num)
        """
        # shape = (target labels num, sentence number, source vectors num)
        W_attention = [[]]*self.target_labels_num

        # j,i means source attribute vector j to target label i
        for i in range(self.target_labels_num):
            # score_pre.shape = (batch size, attributes num, words num)
            score_pre = self.ilp_data[i]['score_pre']
            # attention.shape = (batch size, words number, 1,attribute number*attribute mat size)
            attention = self.ilp_data[i]['attention']
            for l in range(len(score_pre)):
                # shape = (attributes num, words num)
                instance_score_pre = score_pre[l]
                # shape = (words number, 1, attribute number*attribute mat size)
                instance_attention = attention[l]
                for j in range(self.source_vectors_num):
                    # shape = (words num,)
                    source_score = instance_score_pre[j]
                    index = np.argmax(source_score)
                    # shape = (attribute number*attribute mat size,)
                    source_attention = instance_attention[index][0]
                    W_attention[i].append(source_attention)
        # W_attention.shape = (target labels num, source vectors num)
        for i in range(self.target_labels_num):
            W_attention[i] = np.mean(W_attention[i],axis=0)
        # W_attention.shape = (source vectors num, target labels num)
        W_attention = np.transpose(W_attention)
        return W_attention



    def attributes_vec_index(self):
        # W_attention.shape = (source vectors num, target labels num, )
        W_attention = self.extract_attention()
        vars=[[]]*self.source_vectors_num
        for j in range(self.source_vectors_num):
            for i in range(self.target_labels_num):
                vars[j].append(pulp.LpVariable('x_'+str(j)+'_'+str(i),0,1,pulp.LpInteger))
        # space.shape = (source vectors num, target labels num)
        space = np.multiply(W_attention,vars)
        prob = pulp.LpProblem('attr_map',pulp.LpMaximize)

        prob += np.sum(space)
        for j in range(self.source_vectors_num):
            prob+= np.sum(space[j])<=3
        # space.shape = (target labels num, source vectors num)
        space=np.transpose(space)
        for i in range(self.target_labels_num):
            prob+= np.sum(space[i])==3
        prob.solve()

        vars_value=np.zeros(shape=(self.source_vectors_num,self.target_labels_num),dtype='int32')
        for v in prob.variables():
            ls = v.name.split('_')
            j = int(ls[1])
            i = int(ls[2])
            vars_value[j][i]=v.varValue
        index_collection = sorted(np.argwhere(vars_value),key=operator.itemgetter(0,1))
        return index_collection

    def attributes_matrix(self,index_collection,matrix):
        # shape=(target attributes num, mat size, attribute dim)
        A=[[]]*self.target_labels_num
        for index in index_collection:
            j = index[0]
            i = index[1]
            A[i].append(matrix[j])
        return A