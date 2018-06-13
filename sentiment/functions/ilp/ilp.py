import numpy as np
import operator
# import pulp
# import gurobipy

class ILP:
    def __init__(self,depth,layer_dim,used_pos):
        self.depth=depth
        self.lamda = 0.3
        self.delta = 0.3
        self.layer_dim=layer_dim
        self.greatest_k= int(self.layer_dim[-2]/4)
        self.used_pos = used_pos



    def ilp(self, phi_x, W, pos_neg, mt_1):
        """

        :param phi_x: np.narray, shape = (batch size, last hidden layer size)
        :param W: np.array, shape = (last hidden layer size, output layer size)
        :param pos_neg: When at the position where in ground truth label is 1, the pos_neg =1, otherwise, pos_neg =-1
        :param mt_1:
        :return:
        """

        temp=[]
        for i in range(phi_x.shape[0]):
            x = phi_x[i]
            pn = pos_neg[i]
            # print(pn.shape)
            pn = np.tile(np.expand_dims(pn,axis=1),[1,self.layer_dim[-2]])
            t1 = np.multiply(x,np.transpose(W))
            t2 = np.multiply(pn,t1)
            t3 = -np.exp(t2)
            t4 = np.sum(t3,axis=0)
            temp.append(t4)
        z = np.sum(temp,axis=0)

        # abs
        # when mt_1_i = 1, the condition_i = True
        condition = np.equal(np.ones_like(mt_1,dtype='float32'),mt_1)
        abs = np.where(condition, np.ones_like(mt_1,dtype='float32'),-np.ones_like(mt_1,dtype='float32'))
        result = np.add(z,np.multiply(abs,np.multiply(self.lamda,1-mt_1)))
        # rank the result and extract top k
        disordered_result = []
        for i in range(result.shape[0]):
            if i not in self.used_pos:
                disordered_result.append((i,result[i]))
        # order result
        orderd_result = sorted(disordered_result,key=operator.itemgetter(1),reverse=True)
        orderd_result = np.array(orderd_result)
        greatest_k_index = orderd_result[:self.greatest_k,0]
        mt = np.zeros_like(mt_1,dtype='float32')
        for index in greatest_k_index:
            mt[int(index)] = 1
        return mt

    # def matmul(self,vars_mat,constant_mat):
    #     result=[]
    #     for var in vars_mat:
    #         result.append(np.sum(np.multiply(var,constant_mat),axis=1))
    #
    #     return result
    #
    def ilp(self,phi_x, W, pos_neg,mt_1):
        """

        :param phi_x: np.narray
        :param W: np.array
        :param pos_neg:
        :param mt_1:
        :return:
        """
        if self.depth<4:
            upslice=int(self.depth*self.layer_dim[-2]/4)
            lowslice=int((self.depth-1)*self.layer_dim[-2]/4)
            length=np.abs(upslice-lowslice)
        else:
            lowslice=int((self.depth-1)*self.layer_dim[-2]/4)
            length=np.abs(self.layer_dim[-2]-lowslice)
        mask_vars = []
        for i in range(self.layer_dim[-2]):
            name='m'+str(i)
            mask_vars.append(pulp.LpVariable(name,0,1,pulp.LpInteger))
        prob=pulp.LpProblem('',pulp.LpMinimize)
        temp1 = np.multiply(phi_x, mask_vars)
        temp2=[]
        for row in temp1:
            temp2.append(np.sum(np.multiply(row,np.transpose(W)),axis=1))
        temp3 = np.multiply(temp2, pos_neg)

        # manhattan distance
        abs = np.ones_like(mt_1,dtype='float32')
        for i in range(mt_1.shape[0]):
            if mt_1[i]== 1:
                abs[i]=-1
        norm = np.multiply(self.lamda,np.sum(np.add(np.multiply(abs,mask_vars),mt_1)))

        prob += -np.sum(temp3) + norm + self.delta
        # prob+= -np.sum(np.multiply(np.matmul(np.multiply(phi_x,mask_vars),W),pos_neg))+np.linalg.norm(np.subtract(mask_vars,mt_1))
        prob+= np.sum(mask_vars) == length
        prob.solve(pulp.GUROBI_CMD(msg=0))
        mt=[]
        for m in prob.variables():
            num=int(m.name.replace('m',''))
            mt[num]=m.varValue
        return mt