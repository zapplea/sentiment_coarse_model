import numpy as np
import pulp
import operator
# import gurobipy

class AttributeIlp:
    def __init__(self,ilp_data,source_labels_num, mat_size):
        self.ilp_data = ilp_data
        self.attr_mat_size=mat_size
        self.target_labels_num = len(ilp_data)
        self.source_labels_num = source_labels_num
        self.source_vectors_num = source_labels_num*mat_size

    def extract_attention(self):
        """
        extract attention of each sentence
        :return: shape = (target labels num, source vectors num)
        """
        # shape = (target labels num, sentence number, source vectors num)
        W_attention = []

        # j,i means source attribute vector j to target label i
        for i in range(self.target_labels_num):
            # score_pre.shape = (batch size, 1, words num)
            score_pre = self.ilp_data[i]['score_pre']
            # attention.shape = (batch size, words number, 1,attribute number*attribute mat size)
            attention = self.ilp_data[i]['attention']
            tmp=[]
            for l in range(len(score_pre)):
                # shape = (words num, )
                instance_score_pre = score_pre[l][0]
                #print('instance_score_pre_len: ',len(instance_score_pre))
                # shape = (words number, 1, attribute number*attribute mat size)
                instance_attention = attention[l]
                #print('instance_attention_len: ',len(instance_attention))
                #print('source_vectors_num: ',self.source_vectors_num)
                index = np.argmax(instance_score_pre)
                # shape = (attribute number*attribute mat size,)
                source_attention = instance_attention[index][0]
                tmp.append(source_attention)
            W_attention.append(tmp)


        # W_attention.shape = (target labels num, source vectors num)
        for i in range(self.target_labels_num):
            # print('len_',i,':',len(W_attention[i]))
            # print(i,':',np.array(W_attention[i]).shape)
            W_attention[i] = np.mean(W_attention[i],axis=0)
        # W_attention.shape = (source vectors num, target labels num)
        W_attention = np.transpose(W_attention)
        #print('W_attention_shape: ',W_attention.shape)
        return W_attention



    def attributes_vec_index(self):
        # W_attention.shape = (source vectors num, target labels num, )
        W_attention = self.extract_attention()
        # vars.shape = (source vectors num, target labels num)
        vars=[]
        for j in range(self.source_vectors_num):
            tmp = []
            for i in range(self.target_labels_num):
                tmp.append(pulp.LpVariable('x_'+str(j)+'_'+str(i),0,1,pulp.LpInteger))
            vars.append(tmp)
        # space.shape = (source vectors num, target labels num)
        space = np.multiply(W_attention,vars)
        prob = pulp.LpProblem('attr_map',pulp.LpMaximize)

        prob += np.sum(space)
        for j in range(self.source_vectors_num):
            prob+= np.sum(vars[j])<=3
        # space.shape = (target labels num, source vectors num)
        vars_T=np.transpose(vars)
        for i in range(self.target_labels_num):
            prob+= np.sum(vars_T[i])==self.attr_mat_size
        prob.solve()

        index_collection=[]
        for v in prob.variables():
            ls = v.name.split('_')
            j = int(ls[1])
            i = int(ls[2])
            if v.varValue == 1:
                index_collection.append((j,i))
        return index_collection

    def attributes_matrix(self,index_collection,matrix):
        # shape=(target attributes num, mat size, attribute dim)
        A=[]
        for i in range(self.target_labels_num):
            A.append([])
        for index in index_collection:
            j = index[0]
            i = index[1]
            A[i].append(matrix[j])
        return A