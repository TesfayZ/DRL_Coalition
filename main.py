import sys
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from make_task_env import make_env
from Transformation_Function_D import distingusher
import matplotlib.pyplot as plt
import math
import copy
import time
from numpy import savetxt
from numpy import loadtxt
import gurobipy as gp
from gurobipy import GRB 
from TransformerDRL import TDRLAgent, TDRLAgent1q # simultanoues and sequential respectively 
import gc
import os
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'       
EPISODES = 50000
Iterations = 10
TrainAt = "every episode"   # if TrainAt = "every episode" trains at end of episode only, TrainAt="every step" trains at each step
tau = 1  
learning_rate = 0.001 
def MILP(Vs,Ds,Gs,Ws,Ts,h,Bcap):
    #https://docs.python-mip.com/en/latest/examples.html
    if Ts==1:
        H=h # iterations
        L=len(Vs) #total numeber of tasks
        B, I, M = Bcap, range(L), range(H)
        V = [Vs.tolist() for t in M]  
        T = [[Ts for i in I] for t in M]
        G = [Gs.tolist() for t in M]
        W = [Ws.tolist() for t in M] 
        D = [[Ds[i] if G[t][i]<=t<=G[t][i]+W[t][i] else B+1 for i in I] for t in M]  
        B = [B for t in M]
        m = Model()

        x = [[m.add_var(var_type=BINARY) for i in I] for t in M]
        #y = [[m.add_var(var_type=BINARY) for i in I] for t in M]
        #z = [[m.add_var for i in I] for t in Mm]
        #print("x",x[1][1])
        m.objective = maximize(xsum(V[t][i] * x[t][i] for i in I for t in M))
        for t in M:
            m += xsum(D[t][i] * x[t][i] for i in I) <= B[t]
        for i in I:
            m += xsum(x[t][i] for t in M) <= Ts  
        m.optimize()
        selectedk = []
        selectedt = []
        selectedz = []
        best = 0
        for i in I:
            for t in M:
                #if x[t][i].x == 1:
                #selectedk = np.hstack(([selectedk, i]))
                #selectedt = np.hstack(([selectedt, t]))
                #selectedz = np.hstack(([selectedt, z[t+1][i].x]))
                best += V[t][i]*x[t][i].x
                #elif 0<x[t][i].x <1:
                #print("not binary")
        return best
    else:
        #print("This ILP works only for T=1")
        return 0 
def gurobiILP(Vs,Ds,Gs,Ws,Ts,h,Bcap):
    #https://docs.python-mip.com/en/latest/examples.html
    if Ts==1:
        H=h # iterations
        L=len(Vs) #total numeber of tasks
        B, I, M, Mm = Bcap, range(L), range(H), range(H+1)
        V = [Vs.tolist() for t in M]  
        T = [[Ts for i in I] for t in M]
        G = [Gs.tolist() for t in M]
        W = [Ws.tolist() for t in M] 
        D = [[Ds[i] if G[t][i]<=t<=G[t][i]+W[t][i] else Ds[i]+B+1 for i in I] for t in M]  
        B = [B for t in M]
        # Create the Gurobi model
        model = gp.Model()

        # Create the decision variables
        x = {}
        for t in M:
            for i in I:
                x[t,i] = model.addVar(vtype=GRB.BINARY, name=f"x_{t}_{i}")
        # Set the objective function
        model.setObjective(gp.quicksum(V[t][i]*x[t,i] for i in I for t in M), GRB.MAXIMIZE)
        # Add constraints for the maximum weight
        for t in M:
            model.addConstr(gp.quicksum(D[t][i]*x[t,i] for i in I) <= B[t])
        # Add constraints to ensure each item is selected at most once in each time step
        for i in I:
            model.addConstr(gp.quicksum(x[t,i] for t in M) <= 1)

        # Optimize the model
        model.optimize()

        # Print the solution
        return model.objVal
    else:
        #print("This ILP works only for T=1")
        return 0 

if __name__ == "__main__":
    seed = 0 # used only for reproducability of the experiment. Different tasks are generated at different episodes
    if TrainAt not in {"every episode","every step"}:
        print("unknown TrainAt : use 'every episode' or 'every step'")
        exit()
    env = make_env(seed)
    n_agents = 1 # there is only one environment agent with diffferent compy of data for the 4 DRL agents
   
    TransformerDRLstate_size = 5
    TransformerDRLstate_size1q = 4 # sequential doesn not need the distingusher
    TransformerDRLaction_size = 1

    # Define transformer model parameters
    d_model = 8
    num_layers = 6
    num_heads = 8
    ff_dim = 32
    dropout = 0.1
    #create the four agents

    TransformerDRLagent = TDRLAgent(TransformerDRLstate_size, TransformerDRLaction_size, d_model, num_layers, num_heads, ff_dim, dropout, learning_rate)
    TransformerDRLagent1q = TDRLAgent1q(TransformerDRLstate_size1q, TransformerDRLaction_size, d_model, num_layers, num_heads, ff_dim, dropout, learning_rate)
    
    # results variables for TransformerDRL
    TransformerDRLscores, TransformerDRLscores1q = [], []
    TransformerDRLbestscore = 0
    TransformerDRLbestscore1q = 0
    TransformerDRLscounts, TransformerDRLscounts1q, TransformerDRLqcounts, TransformerDRLqcounts1q, TransformerDRLTimes, TransformerDRLTimes1q = [], [], [], [], [], []
    
    #common results
    episodes = []
    TransformerDRLTaskCounts = [] # number of tasks per episode. the same for all of the agents of the same run, no mattter sequenntial, coalition, transformer, 
    for e in range(EPISODES):
        itr=0
        done = False     

        TransformerDRLscore = 0 
        TransformerDRLscore1q = 0
        TransformerDRLqcount = 0
        TransformerDRLqcount1q = 0
        TransformerDRLTime =0.000
        TransformerDRLTime1q = 0.000
        
        #reset environment
        TransformerDRLitems, TransformerDRLBvector, TransformerDRLitems1q, TransformerDRLBvector1q, Bcap, TaskCount, Tmax = env.reset(e, itr)
        while not done: 
      
            # TransformerDRL coalition get_actions
            TransformerDRLtimestart = time.process_time()
            TransformerDRLaction, TransformerDRLprofit, TransformerDRLnextBvector, TransformerDRLnextitems, done, TransformerDRLqcount = TransformerDRLagent.get_action(Tmax, TransformerDRLitems, TransformerDRLBvector, Bcap, TransformerDRLqcount)
            TransformerDRLTime += time.process_time() - TransformerDRLtimestart
            # TransformerDRL sequential get_actions
            TransformerDRLtimestart1q = time.process_time()
            TransformerDRLaction1q, TransformerDRLprofit1q, TransformerDRLnextBvector1q, TransformerDRLnextitems1q, done, TransformerDRLintermediateitems, TransformerDRLintermediateBvectors, TransformerDRLintermediateac, TransformerDRLintermediatenextitems, TransformerDRLintermediatenextBvectors, TransformerDRLintermediatenextac, TransformerDRLqcount1q = TransformerDRLagent1q.get_action(Tmax, TransformerDRLitems1q, TransformerDRLBvector1q, Bcap, TransformerDRLqcount1q)
            TransformerDRLTime1q += time.process_time() - TransformerDRLtimestart1q

            #step 
            itr=itr+1
            TransformerDRLnextitems, TransformerDRLnextBvector, TransformerDRLnextitems1q, TransformerDRLnextBvector1q, Vs, Ds, Gs, Ws, Ts,TaskCount = env.step(TransformerDRLnextitems, TransformerDRLnextBvector, TransformerDRLnextitems1q, TransformerDRLnextBvector1q, e, itr, TaskCount)
  
            if itr==Iterations:
                done = True
            if EPISODES>1 and itr==Iterations:
                done = True
                # compute the benchmark using Integer linear programing : either gurobi or MILP
                ILPgurobi = gurobiILP(Vs, Ds, Gs, Ws, Ts, itr, Bcap)
                ILPMILP = ILPgurobi #MILP(Vs,Ds,Gs,Ws,Ts,itr,Bcap) # uncomment this and the if below to compare the MILP and gurobi based optimizaitons
                '''
                if ILPbest != ILPbest1q:
                    printf("ILPbest != ILPbest1q")
                '''
            #store TransformerDRL simultanous
            TransformerDRLaction = TransformerDRLaction 
            #lexsort items and actions so that it will be easy to detect duplicate in replay memory
            TransformerDRLitems = np.vstack(([np.empty((0,TransformerDRLstate_size), float), TransformerDRLitems, TransformerDRLBvector]))
            TransformerDRLnextactions = np.empty((0,TransformerDRLstate_size), float)
            TransformerDRLnextactions = np.vstack(([TransformerDRLnextactions, TransformerDRLnextitems, TransformerDRLnextBvector]))
            TransformerDRLnextmasks =  np.zeros(len(TransformerDRLnextactions))
            indxs = np.where(TransformerDRLnextactions[:,2] <= TransformerDRLnextBvector[0])# the tasks with demands which can be accepted at the next time slot
            TransformerDRLnextmasks[indxs] = 1
            TransformerDRLnextmasks[-1] = 1 # for the resource constraint vector
            if len(TransformerDRLnextitems)==0:# or np.where(nextitems[:,2] <= nextBvector[0])[0].size==0:
                TransformerDRLagent.append_sample(TransformerDRLitems, TransformerDRLaction, TransformerDRLprofit, TransformerDRLnextactions, True, TransformerDRLnextmasks)
            else:
                TransformerDRLagent.append_sample(TransformerDRLitems, TransformerDRLaction, TransformerDRLprofit, TransformerDRLnextactions, done, TransformerDRLnextmasks)
                            
            # store TransformerDRL sequencial 
            TransformerDRLitems1q = np.vstack(([np.empty((0,4), float), TransformerDRLitems1q, TransformerDRLBvector1q])) 
            TransformerDRLaction1q = TransformerDRLaction1q # no *profit1q because the reward for each item is included in the selection  
            ''''
            if len(TransformerDRLitems1q)!=TransformerDRLaction1q.size:
                print("different",TransformerDRLitems1q,TransformerDRLaction1q)
                exit()
            '''
            TransformerDRLnextactions1q = np.empty((0,4), float)
            TransformerDRLnextactions1q = np.vstack(([TransformerDRLnextactions1q, TransformerDRLnextitems1q]))
            TransformerDRLnextmasks1q =  np.zeros(len(TransformerDRLnextactions1q))
            #print("nextmasks1q",nextmasks1q)
            indxs = np.where(TransformerDRLnextactions1q[:,1] <= TransformerDRLnextBvector1q[0])# the tasks with demands which can be accepted at the next time slot
            TransformerDRLnextmasks1q[indxs] = 1 
            #print("nextmasks1q",nextmasks1q)
            #print("nextactions1q",nextactions1q)
            if len(TransformerDRLnextactions1q)>0:
                if (TransformerDRLnextactions1q[-1,:]==TransformerDRLnextBvector1q).all():
                    TransformerDRLnextmasks1q[-1] = 1 # for the resource constraint vector
            TransformerDRLlenintermediate = len(TransformerDRLintermediateitems)               
            if TransformerDRLlenintermediate==0:
                if len(TransformerDRLnextitems1q)==0:
                    TransformerDRLagent1q.append_sample(TransformerDRLitems1q, TransformerDRLaction1q, TransformerDRLnextactions1q, True, TransformerDRLnextmasks1q)
                else:
                    TransformerDRLagent1q.append_sample(TransformerDRLitems1q, TransformerDRLaction1q, TransformerDRLnextactions1q, done, TransformerDRLnextmasks1q)
            else:
                # now for the intermediates
                i=0
                while i<TransformerDRLlenintermediate-1:
                    TransformerDRLitems1q = np.vstack(([np.empty((0,4), float), TransformerDRLintermediateitems[i]])) 
                    TransformerDRLintermediateaction1q = TransformerDRLintermediateac[i] #  reward for each item is included in the selection 
                    if len(TransformerDRLitems1q)!=TransformerDRLintermediateaction1q.size:
                        print(i,"differentintermedite",TransformerDRLitems1q,TransformerDRLintermediateaction1q)
                        exit()
                    TransformerDRLnextactions1q = np.empty((0,4), float)
                    TransformerDRLnextactions1q = np.vstack(([TransformerDRLnextactions1q, TransformerDRLintermediatenextitems[i]]))
                    TransformerDRLnextmasks1q =  np.zeros(len(TransformerDRLnextactions1q))
                    indxs = np.where(TransformerDRLnextactions1q[:,1] <= TransformerDRLnextBvector1q[0])# 
                    TransformerDRLnextmasks1q[indxs] = 1 
                    if len(TransformerDRLnextactions1q)>0:
                        if (TransformerDRLnextactions1q[-1,:]==TransformerDRLnextBvector1q).all():
                            TransformerDRLnextmasks1q[-1] = 1 # for the noop
                    TransformerDRLagent1q.append_sample(TransformerDRLitems1q, TransformerDRLintermediateaction1q, TransformerDRLnextactions1q, done, TransformerDRLnextmasks1q)
                    i+=1
                #for the last one its next items has mixed newly generated tasks
                TransformerDRLitems1q = np.vstack(([np.empty((0,4), float), TransformerDRLintermediateitems[TransformerDRLlenintermediate-1]])) #because 
                TransformerDRLintermediateaction1q = TransformerDRLintermediateac[TransformerDRLlenintermediate-1] # no *profit1q because the reward for each item is included in the selection  
                TransformerDRLnextactions1q = np.empty((0,4), float)
                TransformerDRLnextactions1q = np.vstack(([TransformerDRLnextactions1q, TransformerDRLnextitems1q]))# because next bvector and items are changed by step.env
                TransformerDRLnextmasks1q =  np.zeros(len(TransformerDRLnextactions1q))
                indxs = np.where(TransformerDRLnextactions1q[:,1] <= TransformerDRLnextBvector1q[0])# 
                TransformerDRLnextmasks1q[indxs] = 1 
                if len(TransformerDRLnextactions1q)>0:
                    if (TransformerDRLnextactions1q[-1,:]==TransformerDRLnextBvector1q).all():
                        TransformerDRLnextmasks1q[-1] = 1 # for the noop
                if TransformerDRLlenintermediate==0:
                    if len(TransformerDRLnextitems1q)==0:
                        TransformerDRLagent1q.append_sample(TransformerDRLitems1q, TransformerDRLintermediateaction1q, TransformerDRLnextactions1q, True, TransformerDRLnextmasks1q)
                    else:
                        TransformerDRLagent1q.append_sample(TransformerDRLitems1q, TransformerDRLintermediateaction1q, TransformerDRLnextactions1q, done, TransformerDRLnextmasks1q)  
            
                          
            TransformerDRLscore += TransformerDRLprofit
            TransformerDRLscore1q += TransformerDRLprofit1q
            TransformerDRLBvector  = TransformerDRLnextBvector
            TransformerDRLBvector1q  = TransformerDRLnextBvector1q
            TransformerDRLitems =  TransformerDRLnextitems
            TransformerDRLitems1q =  TransformerDRLnextitems1q
                   
            if TrainAt == "every step": # Train at each step 
                if TransformerDRLagent.memory.tree.n_entries >= TransformerDRLagent.train_start:# and itr%10==0:
                    TransformerDRLagent.train_model()
                if TransformerDRLagent1q.memory.tree.n_entries >= TransformerDRLagent1q.train_start:# and itr%10==0:
                    TransformerDRLagent1q.train_model()
                # update targets
                if tau == 1:
                    TransformerDRLagent.update_target_model() #tau =1 
                    TransformerDRLagent1q.update_target_model() #tau =1 
                else:
                    TransformerDRLagent.update_network_parameters(tau) # tau =0.01
                    TransformerDRLagent1q.update_network_parameters(tau) # tau =0.01
                   
        if TrainAt == "every episode": # trian only at end of episode
            if TransformerDRLagent.memory.tree.n_entries >= TransformerDRLagent.train_start:# and itr%10==0:
                TransformerDRLagent.train_model()
            if TransformerDRLagent1q.memory.tree.n_entries >= TransformerDRLagent1q.train_start:# and itr%10==0:
                TransformerDRLagent1q.train_model()
            # update targets
            if tau == 1:
                TransformerDRLagent.update_target_model() #tau =1 
                TransformerDRLagent1q.update_target_model() #tau =1 
            else:
                TransformerDRLagent.update_network_parameters(tau) # tau =0.01
                TransformerDRLagent1q.update_network_parameters(tau) # tau =0.01 
        
        TransformerDRLscores.append(TransformerDRLscore/ILPgurobi)
        TransformerDRLscores1q.append(TransformerDRLscore1q/ILPgurobi)

        
        TransformerDRLTaskCounts.append(TaskCount)
        #TransformerDRLscounts.append(TransformerDRLscount)  # transformer has no state count
        #TransformerDRLscounts1q.append(TransformerDRLscount1q)
        TransformerDRLqcounts.append(TransformerDRLqcount)
        TransformerDRLqcounts1q.append(TransformerDRLqcount1q)
        TransformerDRLTimes.append(TransformerDRLTime)
        TransformerDRLTimes1q.append(TransformerDRLTime1q)
        
        episodes.append(e)
        
        #saving to file
        nran= sys.argv[1]# index of the run for saving in csv file = 00
        
        #update the csv files after every after 100 episodes
        if e%1000 ==0:
            #saving TransformerDRLresults
            TransformerDRLsavescores = np.array(TransformerDRLscores)
            savetxt('./TransformerDRLresults/rMLP'+str(nran)+'savescores.csv', TransformerDRLsavescores)
            TransformerDRLsavescores1q = np.array(TransformerDRLscores1q)
            savetxt('./TransformerDRLresults/rMLP'+str(nran)+'savescores1q.csv', TransformerDRLsavescores1q)
            TransformerDRLsaveTaskCounts = np.array(TransformerDRLTaskCounts)
            savetxt('./TransformerDRLresults/rMLP'+str(nran)+'saveTaskCounts.csv', TransformerDRLsaveTaskCounts) 
            #TransformerDRLsavescounts = np.array(TransformerDRLscounts)
            #savetxt('./TransformerDRLresults/rMLP'+str(nran)+'savescounts.csv', TransformerDRLsavescounts) 
            #TransformerDRLsavescount1qs = np.array(TransformerDRLscounts1q)
            #savetxt('./TransformerDRLresults/rMLP'+str(nran)+'savescount1qs.csv', TransformerDRLsavescount1qs) 
            TransformerDRLsaveqcounts = np.array(TransformerDRLqcounts)
            savetxt('./TransformerDRLresults/rMLP'+str(nran)+'saveqcounts.csv', TransformerDRLsaveqcounts) 
            TransformerDRLsaveqcount1qs = np.array(TransformerDRLqcounts1q)
            savetxt('./TransformerDRLresults/rMLP'+str(nran)+'saveqcounts1q.csv', TransformerDRLsaveqcount1qs) 
            TransformerDRLsaveTimes = np.array(TransformerDRLTimes)
            savetxt('./TransformerDRLresults/rMLP'+str(nran)+'saveTimes.csv', TransformerDRLsaveTimes) 
            TransformerDRLsaveTimes1q = np.array(TransformerDRLTimes1q)
            savetxt('./TransformerDRLresults/rMLP'+str(nran)+'saveTimes1q.csv', TransformerDRLsaveTimes1q) 
            

            
    
            
        '''
        # save the model if it scores better averate 50 episodes than the prevous score
        if 
            #print("Models saved")
            torch.save(agent.model, "./rMLP"+str(nran)+"Mul")
            torch.save(agent1q.model, "./rMLP"+str(nran)+"Seq")
        '''
        

          
          
