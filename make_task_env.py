import random
import copy
import math
import numpy as np
from Transformation_Function_D import distingusher
def make_env(seed=None):
    env = node_agent_environment(1, seed)# 1 environment agent
    return env
class node_agent_environment:
    def __init__(self, n_env_agents, seed = None):
        self.n = n_env_agents        
        self.agents = [node_agent() for i in range(self.n)]
        self.Tmax = 1 # time to stay occupying bandwidth
        self.MaxTasks=10 # maximum number of tasks to be generated at once
        self.Wmax = 3 # waiting time before allocation
        self.Vmax=5 # max value of the tasks
        self.Bcap= 100 # bandwidth capacity
        self.Demands= [8,32,64] # bandwidth demand of tasks
        if seed is not None:
            random.seed(seed)  # Set the random seed if provided
    def step(self, TransformerDRLitems, TransformerDRLnextBvector, TransformerDRLitems1q, TransformerDRLnextBvector1q,  e, itr, TaskCount):
        for i, agent in enumerate(self.agents):
                     
            ##### TransformerDRL simultanous deacrease waiting time
            TransformerDRLitems[:, 3] -= 1
            exp = np.where(TransformerDRLitems[:,3]< 0)[0]
            expsize = exp.size
            if expsize>0:
                TransformerDRLitems = np.delete(TransformerDRLitems, exp, 0)
            agent.TransformerDRLTasklist = TransformerDRLitems
            agent.TransformerDRLBvector = TransformerDRLnextBvector
            
            
            ##### TransformerDRL sequencial deacrease waiting time
            TransformerDRLitems1q[:, 2] -= 1
            exp1q = np.where(TransformerDRLitems1q[:,2]< 0)[0]
            expsize1q = exp1q.size
            if expsize1q>0:
                TransformerDRLitems1q = np.delete(TransformerDRLitems1q, exp1q, 0)
            agent.TransformerDRLTasklist1q = TransformerDRLitems1q
            agent.TransformerDRLBvector1q = TransformerDRLnextBvector1q 
            
            newTasks = random.randint(0, self.MaxTasks+1)
            TaskCount += newTasks
            self.generateTasks(agent, i, newTasks, e, itr)
            
            # update resource vector
            for j in range(agent.TransformerDRLBvector[0].size-1):
                agent.TransformerDRLBvector[j] = agent.TransformerDRLBvector[j+1]
            for j in range(agent.TransformerDRLBvector1q[0].size-1):
                agent.TransformerDRLBvector1q[j] = agent.TransformerDRLBvector1q[j+1]
            agent.TransformerDRLBvector[agent.TransformerDRLBvector[0].size-1] = self.Bcap
            agent.TransformerDRLBvector1q[agent.TransformerDRLBvector1q[0].size-1] = self.Bcap 
        return agent.TransformerDRLTasklist, self.agents[0].TransformerDRLBvector, agent.TransformerDRLTasklist1q, self.agents[0].TransformerDRLBvector1q, agent.Vs, agent.Ds, agent.Gs, agent.Ws, self.Tmax, TaskCount
        
    def reset(self, e, itr):# call generate tasks for all agents
        for idx, agent in enumerate(self.agents):
            Time =0
            Time1q =0
            agent.TransformerDRLTasklist = np.empty((0,5), float)
            agent.TransformerDRLTasklist1q = np.empty((0,4), float)
            agent.Vs = []
            agent.Ds = []
            agent.Gs = []
            agent.Ws = []
            newTasks = self.MaxTasks
            self.generateTasks(agent, idx, newTasks, e, itr)
            # it must be padded to be equal size with the tasks so that the transformer uses the same embeding dimention
            shape_tuple = agent.TransformerDRLTasklist.shape
            agent.TransformerDRLBvector = np.ones(shape_tuple[1])*self.Bcap
            shape_tuple = agent.TransformerDRLTasklist1q.shape
            agent.TransformerDRLBvector1q = np.ones(shape_tuple[1])*self.Bcap
        return self.agents[0].TransformerDRLTasklist, self.agents[0].TransformerDRLBvector, self.agents[0].TransformerDRLTasklist1q, self.agents[0].TransformerDRLBvector1q, self.Bcap, newTasks, self.Tmax
    
    def generateTasks(self, object, idx, newTasks, e, itr):
        for i in range(newTasks):
            Tvalue = random.randint(1,self.Vmax)
            Tdemand = random.choice(self.Demands)
            Tw = random.randint(1,self.Wmax)
            Tt = random.randint(1,self.Tmax)
            
            TransformerDRLtempTask=np.array([1, Tvalue, Tdemand, Tw, Tt])
            TransformerDRLtempTask1q=np.array([Tvalue, Tdemand, Tw, Tt]) # no need of distingusher for sequencial
            
            object.TransformerDRLTasklist=np.vstack(([object.TransformerDRLTasklist,TransformerDRLtempTask])) # 
            object.TransformerDRLTasklist1q=np.vstack(([object.TransformerDRLTasklist1q,TransformerDRLtempTask1q])) #
            
            object.Vs=np.hstack(([object.Vs, Tvalue]))
            object.Ds=np.hstack(([object.Ds, Tdemand]))
            object.Gs=np.hstack(([object.Gs, itr]))
            object.Ws=np.hstack(([object.Ws, Tw]))
        object.TransformerDRLTasklist = distingusher(object.TransformerDRLTasklist)
        #object.TransformerDRLTasklist1q = distingusher(object.TransformerDRLTasklist1q) because no need of distingusher for sequencial
class node_agent():
    def __init__(self):
        self.TransformerDRLBvector= []
        self.TransformerDRLTasklist = []
        self.TransformerDRLBvector1q= []
        self.TransformerDRLTasklist1q = []
        
        # for gurobi or MLIP optimization
        self.Vs = []
        self.Ds = []
        self.Gs = []
        self.Ws = []
        

        
        
