import random
import logging
import math
import numpy as np

class SpecEnv_rankOpt(object):
    def __init__(self, NAP, NUE, NTRAIN, NTEST, interdist=100):
        # self.seed()
        self.NAP = NAP
        self.NUE = NUE
        self.NTEST = NTEST
        self.NTRAIN = NTRAIN
        self.intd = interdist
        self.y_data = np.zeros([self.NTRAIN,self.NAP,self.NUE])	# 
        self.x_data = np.zeros([self.NTRAIN,self.NAP,self.NUE])	# distance to each AP

        self.y_test = np.zeros([self.NTEST,self.NAP,self.NUE])	# 
        self.x_test = np.zeros([self.NTEST,self.NAP,self.NUE])	# distance to each AP

        self.root = int(self.NAP**.5)
        self.AP = [(i+self.intd,j+self.intd) for i in range(0,self.root*(self.intd),self.intd) for j in range(0,self.root*(self.intd),self.intd)]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def topoInit(self):
        """Example:
        NAP = 4
        interdist = 100
        traffic_rate = 1
        """
        # Training Dataset
        tmp = self.AP
        for t in range(self.NTRAIN):
            for nu in range(self.NUE):
                thisDist = np.random.rand(2)*(self.root+2)*(self.intd)
                dall = []
                for na in range(self.NAP):
                    thisD = ((tmp[na][0]-thisDist[0])**2 + (tmp[na][1]-thisDist[1])**2)**(.5)
                    self.x_data[t][na][nu] = thisD
                    dall.append(thisD)
                # print(dall)
                idx1 = np.argsort(np.array([dall]))
                # print(idx1[0][0])
                # print(idx1[0][1])
                # print(idx1)
                self.y_data[t][idx1[0][0]][nu] = 1
                # self.y_data[t][idx1[0][1]][nu] = 1
        # Testing Dataset
        for t in range(self.NTEST):
            for nu in range(self.NUE):
                thisDist = np.random.rand(2)*(self.root+2)*(self.intd)
                dall = []
                for na in range(self.NAP):
                    thisD = ((tmp[na][0]-thisDist[0])**2 + (tmp[na][1]-thisDist[1])**2)**(.5)
                    self.x_test[t][na][nu] = thisD
                    dall.append(thisD)
                idx1 = np.argsort(np.array([dall]))
                self.y_test[t][idx1[0][0]][nu] = 1
                # self.y_test[t][idx1[0][1]][nu] = 1

        return self.x_data,self.y_data,self.x_test,self.y_test

    
    def UMI(self,dist):
        if dist>0:
            dB = 30.18 + 26.7*math.log10(dist)
            return 10**(-dB/10)
        else:
            dB = 30.18 + 26.7*math.log10(30)
            return 10**(-dB/10)

    def delayCalc(self):
        rate = {i:0 for i in range(self.NAP)}
        for i in range(self.NAP):
            tmp = self.state[i]
            for j in range(len(tmp)):
                if tmp[j] == 1:
                    interf = 0+self.noise
                    for p in range(self.NAP):
                        if self.state[p][j] == 1 and p!=i:
                            interf += self.PL[i][p]
                    tmp_rate = 0.18/0.5*math.log10(1+min(1000,self.PL[i][i]/interf))/math.log10(2)
                    rate[i] = rate.get(i,0) + tmp_rate
        # print(rate)
        sum_LogRate = 0
        delay = 0
        for key in rate:
            # sum_LogRate += math.log10(rate[key])
            # sum_LogRate += rate[key]
            if rate[key]<=self.traffic[key]:
                delay += 1000
            else:
                delay += (10*self.traffic[key]/(rate[key]-self.traffic[key]))**2
        return -delay/sum(self.traffic)
        # return sum_LogRate*100

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)
    
    def step(self,action):
        # print(action)
        thisAction = np.reshape(action,(self.NAP,self.NSPEC))
        # thisAction = np.clip(thisAction,-0.99,0.99)/2+0.5   # tanh
        thisAction = np.clip(thisAction,0,1)   # sigmoid
        thisAction = np.around(thisAction)
        # if np.max(thisAction)==0:
        #     thisAction[0] = 1
        self.state = thisAction
        reward = self.delayCalc()
        # if np.max(thisAction)==0:
        #     reward = -100000000
        # if reward>-1e-10:
        #     done = True
        # else:
        #     done = False
        done = False
        # Add delay constraint into observation
        thisObservation = np.reshape(thisAction,self.NAP*self.NSPEC)
        return thisObservation, reward, done, {}
                

    def reset(self):
        thisObservation = np.ones(self.NAP*self.NSPEC)
        self.state = np.ones((self.NAP,self.NSPEC))

        # Random Initialization
        # thisObservation = np.random.randint(2,size=self.NAP*self.NSPEC)
        # self.state = np.random.randint(2,size=(self.NAP,self.NSPEC))
        return thisObservation

    def render(self, mode='human', close=False):

        return 'Finished'



# NAP = 9
# NUE = 30
# interdist = 100
# NTRAIN = 10000
# NTEST = 1000
# env = SpecEnv_rankOpt(NAP, NUE, NTRAIN, NTEST, interdist)
# # the data, shuffled and split between train and test sets
# x_train, y_train, x_test, y_test = env.topoInit()




