#from methods_and_benchmaks import LexTrans, LexTrans2
#from NumericalEncoder import NumericalTransformer
from TransformerEncoder  import Encoder
import sys
import torch
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from prioritized_memory import Memory
from Transformation_Function_D import distingusher
import matplotlib.pyplot as plt
import math
import copy
import time
from numpy import savetxt
from numpy import loadtxt
import gurobipy as gp
from gurobipy import GRB
            
EPISODES = 50000
Iterations = 10
load =False
class TDRLAgent():# Coalition
    def __init__(self, input_size, action_size, d_model, num_layers, num_heads, ff_dim, dropout, learning_rate = 0.001):
        self.tau = 0.01
        # get size of state and action
        self.state_size = input_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = learning_rate
        self.memory_size = 10000
        self.epsilon = 1.0
        self.epsilon_min = 0
        self.explore_step = 5000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.batch_size = 64
        self.train_start = 1000

        # create prioritized replay memory using SumTree
        self.memory = Memory(self.memory_size)

        # create main model and target model
        #self.model = DQN(state_size, action_size)
        self.model = Encoder(self.state_size, self.action_size , d_model, num_layers, num_heads, ff_dim, dropout)
        self.model.apply(self.weights_init)
        #self.target_model = DQN(state_size, action_size)
        self.target_model = Encoder(self.state_size, self.action_size , d_model, num_layers, num_heads, ff_dim, dropout)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)

        # initialize target model
        self.update_target_model()

    # weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform_(m.weight)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    # another option with different update rate
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_model_params = self.target_model.named_parameters()
        model_params = self.model.named_parameters()

        target_model_state_dict = dict(target_model_params)
        model_state_dict = dict(model_params)
        for name in model_state_dict:
            model_state_dict[name] = tau*model_state_dict[name].clone() + \
                    (1-tau)*target_model_state_dict[name].clone()

        self.target_model.load_state_dict(model_state_dict)
    # get action from model using epsilon-greedy policy
    def get_action(self,  Tmax, items, Bvector, Bcap, qcount):
        self.model.eval()
        self.target_model.eval()
        orindxs = np.where(items[:,2] <= Bvector[0])# the tasks with demands which can be accepted at the current time slot
        indxs=orindxs[0]
        left =  np.where(items[:,2]> Bvector[0])[0]# tasks with greater demand than capacity
        left = left.size
        le = indxs.size
        done = False
        total = le+left
        ac = np.zeros(total+1) # +1 because the resource is included constraint 
        selectedlist = []
        reward = 0
        indxs = np.hstack([indxs, total])
        '''
        print("Before")
        print("Bvector", Bvector)
        print("Items", items)
        '''
        if le>0: # if there is at least one customer to be served
            if np.random.rand() <= self.epsilon:
                #print("le",le)
                #randomorder = random.sample(range(le),le)
                randomorder = random.sample(indxs.tolist(), indxs.size)
                for i in range(indxs.size):
                    selectedT =  randomorder[i]
                    if selectedT == total:
                        pass # it is the Bvector
                    elif Bvector[0] >= items[selectedT,2]:
                        reward += items[selectedT,1]
                        ac[selectedT] = 1
                        #print("items", items)
                        for v in range(int(items[selectedT,4])): # for the duration of the task
                            #print("Bvector", Bvector)
                            Bvector[v] -= items[selectedT,2] # subtract the demand from the capacity, +1 because of the distinuisher
                        selectedlist.append(selectedT)
                items = np.delete(items,selectedlist,0)
                '''
                print("After")
                print("Bvector", Bvector)
                print("Items", items)
                print("ac", ac)
                '''
                return ac, reward, Bvector, items, done, qcount
            else:
                Qs=[]
                titems=items#[indxs[:le]]
                #print("titems",titems)
                #print("Bvector",Bvector)
                #print("stack",np.vstack([titems, Bvector]))
                Qs = self.model(Variable(torch.from_numpy(np.vstack([titems, Bvector])).float().unsqueeze(0)), None)
                qcount += len(np.vstack([titems, Bvector]))
                #print("Qs",Qs)
                _, idd = torch.sort(Qs, dim=1, descending=True)# torch.argsort(-1*torch.stack(Qs,-1))# sort in decreasing order
                #print("Qs",idd)
                for i in range(ac.size):
                    selectedT =  idd[0][i][0]
                    if selectedT == total:
                        pass # it is the Bvector
                    elif Bvector[0] >= items[selectedT,2]:
                        reward += items[selectedT,1]
                        ac[selectedT] = 1
                        for v in range(int(items[selectedT,4])): # for the duration of the task
                            Bvector[v] -= items[selectedT,2] # subtract the demand from the capacity
                        selectedlist.append(selectedT)
                items = np.delete(items,selectedlist,0)
                '''
                print("After")
                print("Bvector", Bvector)
                print("Items", items)
                print("ac", ac)
                '''
                return ac, reward, Bvector, items, done, qcount
        #else: 
            #reward = 0 
            #done = True
        
        print("After")
        print("Bvector", Bvector)
        print("Items", items)
        print("ac", ac)
        
        return ac, reward, Bvector, items, done, qcount

    # save sample (error,<s,a,r,s'>) to the replay memory
    def append_sample(self, items, actions, profit, nextitems, done, nextmasks):
        self.model.eval()
        self.target_model.eval()
        error = 0
        rewards = np.zeros_like(actions)                                 
        if len(items)>0:                                
            olds = self.model(Variable(torch.FloatTensor(items)), None).data
            #print("olds",olds) 
            targets_val = np.zeros_like(olds)
            if len(nextitems)>0:
                nextmasks = torch.Tensor(nextmasks)
                targets = self.target_model(Variable(torch.FloatTensor(nextitems)), None).data*nextmasks.reshape(len(nextitems),1)
                #print("nextitems",nextitems)
                #print("targets",nextmasks)
                discountedtarget = self.discount_factor * torch.max(targets)
            for i in range(len(items)):                             
                if actions[i]>0: # store only if reward is not zero
                    rewards[i] = profit
                    if i==len(actions)-1:# for the noop
                        rewards[i] = rewards[i]-2
                    if done or len(nextitems)==0:
                        targets_val[i] = rewards[i]
                    else:
                        targets_val[i] = rewards[i] + discountedtarget
                    error += torch.sum(abs(olds[i] - targets_val[i])).item()
                    if isinstance(error, np.ndarray):
                        error =  np.sum(error)
            #print("error", error)
            self.memory.addorupdate(error, (items, actions, rewards, nextitems, nextmasks, done))

    # pick samples from prioritized replay memory (with batch_size)
    def train_model(self):
        self.model.train()
        self.target_model.train()
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            if self.epsilon<0:
                self.epsilon=0
        tryfetch = 0
        while tryfetch<3:
          mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
          #print("idxs, is_weights", len(idxs), len(is_weights))
          mini_batch = np.array(mini_batch, dtype=object).transpose()
          if any(not isinstance(arr, np.ndarray) for arr in mini_batch[0]) or any(not isinstance(arr, np.ndarray) for arr in mini_batch[2]):
              if tryfetch<3:
                  tryfetch += 1
              else:
                  print("mini_batch = ", mini_batch)
                  exit()
          else:
              break
        items = list(mini_batch[0]) 
        #print("items", items)
        max_lengthit = max(len(arr) for arr in items)
        #print("max_lengthit", max_lengthit)
        tempitems = np.zeros((len(items), max_lengthit, self.state_size))
        for i, sequence in enumerate(items):
            tempitems[i, :len(sequence), :] = sequence
        #tempitems = [np.pad(item, ((0, max_lengthit - len(item)))) for item in items]
        items = tempitems#np.vstack(tempitems)       
        #print("itemslen", len(items))
        actions = list(mini_batch[1])  
        #print("actions", actions)
        max_lengthac = max(arr.size for arr in actions)
        #print("max_lengthac", max_lengthac)
        hidenmasks = copy.deepcopy(actions)
        hidenmasks = [np.where(hidenmask == 0, 1, hidenmask) for hidenmask in hidenmasks]
        tempactions = []
        temphidenmasks = []
        tempactions = [np.pad(item, ((0, max_lengthac - item.size))) for item in actions]
        temphidenmasks = [np.pad(item, ((0, max_lengthac - item.size))) for item in hidenmasks]
        #print("tempactions", tempactions)
        actions = np.vstack(tempactions)
        #print("actions", actions)
        hidenmasks = np.vstack(temphidenmasks)
        #print("actions", actions)
                                         
        rewards = list(mini_batch[2])
        max_lengthrw = max(arr.size for arr in rewards)
        temprewards = [np.pad(item, ((0, max_lengthrw - item.size))) for item in rewards]    
        rewards = np.vstack(temprewards)
        currentmasks = actions # not necessary becuase the action is used to mask pred below                                
        #print("rewards", len(rewards))                            
        nextitems = list(mini_batch[3]) 
        #print("nexitems", len(nextitems))
        nextmasks = list(mini_batch[4])                                
        dones = list(mini_batch[5])

        if max_lengthac!= max_lengthrw != max_lengthit:
            print("batchsize mismatch")
            #exit()   
                                
        # bool to binary
        dones = np.array(dones).astype(int)

        # Q function of current state
        states = torch.Tensor(items)
        states = Variable(states).float()
        #print("items", states.shape)
        actions = torch.FloatTensor(actions)
        hidenmasks = torch.FloatTensor(hidenmasks)
        pred = self.model(states, hidenmasks)
        #print("actions", actions.shape)
        #total loss of the seleced actions at every batch using the action as masking
        #pred = torch.sum(pred.mul(Variable(actions)), dim=1)
        #pred = torch.sum(pred.mul(actions.unsqueeze(-1)), dim=1)   
        pred = pred.mul(actions.unsqueeze(-1)) 
        #print("pred", pred.shape)
        # Q function of next state
        next_pred = torch.zeros(self.batch_size, 1)#The size of next states mast match the current states
        for n in range(self.batch_size): 
            if  len(nextitems[n])>0:
                inext_states = torch.Tensor(nextitems[n])
                inext_states = Variable(inext_states).float().unsqueeze(0)  
                inextmasks = nextmasks[n]
                inextmasks = torch.FloatTensor(inextmasks)
                next_pred[n] = torch.max(self.target_model(inext_states, None).data*inextmasks.reshape(len(nextitems[n]),1))
        #print("next_pred", next_pred)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        # Q Learning: get maximum Q value at s' from target model
        #print("actions", actions)
        #print("rewards", rewards)
        rewards = rewards.mul(Variable(actions))  # becuase we don't need reward and nexte states for the unselected  ones 
        dones = dones.reshape(self.batch_size, 1)
        #print("rewards", rewards.shape)
        #print("dones", dones)
        discounted_next_pred = (1 - dones) * self.discount_factor * next_pred
        #print("discountednextpred", discountednextpred)
        #print("shepeddiscountednextpred", actions * discounted_next_pred)
        target = rewards + actions * discounted_next_pred# masking the next pred with the actions, reward is already masked
        target = Variable(target)
        #print("target", target.shape)
        #target = torch.sum(target.mul(Variable(actions)), dim=1)  # becuase we don't need reward and nexte states for the unselected  ones                   
        #print("actions", actions)
        errors = torch.abs(pred.squeeze(dim=-1) - target).data.numpy()

        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            error = errors[i]
            if isinstance(error, np.ndarray):
                error =  np.sum(error)
            #print("errors",idx,errors[i])
            self.memory.update(idx, error)

        self.optimizer.zero_grad()

        # MSE Loss function
        loss = (torch.FloatTensor(is_weights) * F.mse_loss(pred.squeeze(dim=-1), target)).mean()
        #loss =  F.mse_loss(pred, target)
        loss.backward()

        # and train
        self.optimizer.step()  
        
class TDRLAgent1q():#sequential 
    def __init__(self, input_size, action_size, d_model, num_layers, num_heads, ff_dim, dropout, learning_rate = 0.001):
        self.tau = 0.01
        # get size of state and action
        self.state_size = input_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = learning_rate = 0.001
        self.memory_size = 10000
        self.epsilon = 1.0
        self.epsilon_min = 0
        self.explore_step = 5000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.batch_size = 64
        self.train_start = 1000

        # create prioritized replay memory using SumTree
        self.memory = Memory(self.memory_size)

        # create main model and target model
        #self.model = DQN(state_size, action_size)
        self.model = Encoder(input_size, action_size , d_model, num_layers, num_heads, ff_dim, dropout)
        self.model.apply(self.weights_init)
        #self.target_model = DQN(state_size, action_size)
        self.target_model = Encoder(input_size, action_size, d_model, num_layers, num_heads, ff_dim, dropout)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)

        # initialize target model
        self.update_target_model()


    # weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform_(m.weight)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    # another option with different update rate
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_model_params = self.target_model.named_parameters()
        model_params = self.model.named_parameters()

        target_model_state_dict = dict(target_model_params)
        model_state_dict = dict(model_params)
        for name in model_state_dict:
            model_state_dict[name] = tau*model_state_dict[name].clone() + \
                    (1-tau)*target_model_state_dict[name].clone()

        self.target_model.load_state_dict(model_state_dict)
    # get action from model using epsilon-greedy policy
    def get_action(self, Tmax, items, Bvector, Bcap, qcount1q):
        self.model.eval()
        self.target_model.eval()
        qcount1q = qcount1q
        indxs = np.where(items[:,1] <= Bvector[0])# the tasks with demands which can be accepted at the current time slot
        #print("indxs",indxs)
        #print("Bvector[0]",Bvector[0])
        indxs=indxs[0]
        left =  np.where(items[:,1]> Bvector[0])[0]# tasks with greater demand than capacity
        left = left.size
        le = indxs.size
        done = False
        reward = 0
        originalitems = copy.deepcopy(items)
        selectedList = []
        deleteList = []
        intermediateitems, intermediateBvectors, intermediateac, intermediatenextitems, intermediatenextBvectors, intermediatenextac = [], [], [], [], [], []
        total = le+left
        ac = np.zeros(total+1) # because there is noop
        indxs = np.hstack([indxs, total]) # assuming noop is at the end
        if np.random.rand() <= self.epsilon:
            #print("explore")
            #print("items before",le,originalitems )
            while le>0: # if there is at least one customer to be served,
                ac = np.zeros(total+1)
                #print("le",le)
                #randomorder = random.sample(range(le),1)
                selectedT = random.choice(indxs)
                #print("selectedT",selectedT)
                #selectedT =  indxs[randomorder]
                if selectedT == total:
                    indxs = np.setdiff1d(indxs, selectedT) 
                    le = indxs.size
                    selectedList.append(selectedT)
                    pass
                    '''
                    ac[selectedT] = -1
                    intermediateac.append(np.delete(ac,selectedList))
                    intermediateitems.append(np.delete(items,selectedList,0))
                    intermediateBvectors.append(Bvector)
                    #print("ac and items", np.delete(ac,selectedList), np.delete(items,selectedList,0))
                    break
                    '''
                elif Bvector[0] >= originalitems[selectedT,1]:
                    reward += originalitems[selectedT,0]
                    ac[selectedT] = originalitems[selectedT,0]
                    ac[-1]=0
                    intermediateac.append(np.delete(ac,selectedList))# delete the previous selected lists so that only the current action is 1 before it is sent to memory
                    intermediateitems.append(np.delete(np.vstack([originalitems,Bvector]),selectedList,0))
                    intermediateBvectors.append(Bvector)
                    #print("ac and items", np.delete(ac,selectedList), np.delete(items,selectedList,0))
                    #print("items",originalitems )
                    for v in range(int(originalitems[selectedT,3])): # for the duration of the task
                        Bvector[v] -= originalitems[selectedT,1] # subtract the demand from the capacity
                    selectedList.append(selectedT)# now include the selected action in list of selected so that it helps to remove others next 
                    deleteList.append(selectedT)
                    #print("itemss and selectedT", items, selectedT )
                    #xxx, state = LexTrans2(1, np.empty((0,5), float), 0, distingusher(np.delete(items,selectedList,0)), Bvector, Bcap, Tmax, 0, 0)

                    left =  np.where(originalitems[:,1]> Bvector[0])[0]
                    indxs = np.setdiff1d(indxs, selectedT) 
                    indxs = np.setdiff1d(indxs, left) 
                    le = indxs.size
                    intermediatenextac.append(np.delete(ac,selectedList))
                    intermediatenextitems.append(np.delete(np.vstack([originalitems,Bvector]),selectedList,0))
                    intermediatenextBvectors.append(Bvector)
                '''
                else:
                    indxs = np.setdiff1d(indxs, selectedT) 
                    le = indxs.size
                '''
        else:
            #print("================================exploit======================")
            while le>0: # if there is at least one customer to be served,
                ac = np.zeros(total+1)
                titems = np.vstack([originalitems, Bvector])
                allindxs = np.arange(len(titems))
                Qs = np.zeros(allindxs.size)-1000.989# to make it float not int
                #print("Qs",Qs)
                ititems = np.delete(titems, selectedList,0) 
                
                #print("indxs",indxs, allindxs, selectedList)
                #print("Bvector",Bvector)
                #print("stack",np.vstack([titems, Bvector]))
                tQs = self.model(Variable(torch.from_numpy(ititems).float().unsqueeze(0)), None)
                #print("tQs",tQs)
                
                Qs[np.setdiff1d(allindxs, selectedList)] = tQs[0,:,0].detach().numpy()#[np.setdiff1d(allindxs, selectedList)]
                Qs = Qs[indxs]
                #print("Qs",Qs)
                qcount1q += len(ititems) # the number of executed tasks
                #_, idd = torch.sort(Qs, dim=1, descending=True)# torch.argsort(-1*Qs)# sort in decreasing order
                idd = np.argsort(Qs)[::-1]
                #for i in range(le):
                #print("idd",idd, indxs)
                selectedT = indxs[idd[0]]  #indxs[idd[0][0][0]]
                #print("selected and total", selectedT, total)
                if selectedT == total:
                    #noop
                    indxs = np.setdiff1d(indxs, selectedT) 
                    le = indxs.size
                    selectedList.append(selectedT)
                    '''
                    ac[selectedT] = -1
                    intermediateac.append(np.delete(ac,selectedList))
                    intermediateitems.append(np.delete(items,selectedList,0))
                    intermediateBvectors.append(Bvector)
                    #print("ac and items", np.delete(ac,selectedList), np.delete(items,selectedList,0))
                    break
                    '''
                elif Bvector[0] >= originalitems[selectedT,1]:
                    reward += originalitems[selectedT,0]
                    ac[selectedT] = originalitems[selectedT,0]
                    ac[-1]=0
                    intermediateac.append(np.delete(ac,selectedList))
                    intermediateitems.append(np.delete(np.vstack([items,Bvector]),selectedList,0))
                    intermediateBvectors.append(Bvector)
                    #print("ac and items", np.delete(ac,selectedList), np.delete(items,selectedList,0))
                    for v in range(int(originalitems[selectedT,3])): # for the duration of the task
                        Bvector[v] -= originalitems[selectedT,1] # subtract the demand from the capacity
                    selectedList.append(selectedT)
                    deleteList.append(selectedT)
                    left =  np.where(items[:,1]> Bvector[0])[0]
                    indxs = np.setdiff1d(indxs, selectedT) 
                    indxs = np.setdiff1d(indxs, left) 
                    le = indxs.size
                    intermediatenextac.append(np.delete(ac,selectedList))
                    intermediatenextitems.append(np.delete(np.vstack([items,Bvector]),selectedList,0))
                    intermediatenextBvectors.append(Bvector)
                '''
                else:
                    indxs = np.setdiff1d(indxs, selectedT) 
                    le = indxs.size
                '''    
        ac[-1]=0
        return ac, reward, Bvector, np.delete(items,deleteList,0), done, intermediateitems, intermediateBvectors, intermediateac, intermediatenextitems, intermediatenextBvectors, intermediatenextac, qcount1q

    # save sample (error,<s,a,r,s'>) to the replay memory
    def append_sample(self, items, action, nextitems, done, nextmasks):
        self.model.eval()
        self.target_model.eval()
        #agent1q.append_sample(items1q, action1q, nextactions1q, done, nextmasks1q)
        error = 0
        actions = (action != 0).astype(int)                                
        rewards = action 
        '''
        print("samplea",action)
        print("sampleas",actions)
        print("sampler",rewards)
        '''
        if len(items)>0:
            if len(items)==1 or len(items)==items.size==4: # to identify if the list has only one item
                items = np.reshape(items, (1, 4))
                #print("np.reshape(items, (1, 4)))",items)
            olds = self.model(Variable(torch.FloatTensor(items)), None).data
            targets_val = np.zeros_like(olds)
            if len(nextitems)>0:
                nextmasks = nextmasks = torch.Tensor(nextmasks)
                targets = self.target_model(Variable(torch.FloatTensor(nextitems)), None).data*nextmasks.reshape(len(nextitems),1)
                discountedtarget = self.discount_factor * torch.max(targets)
            for i in range(len(items)):
                #print("action",action)                              
                if action[i]!=0: # store only if reward is not zero
                    #print("action[i]",action[i])
                    if action[i]==-1:# for the noop
                        rewards[i] = 0
                    if done or len(nextitems)==0:
                        targets_val[i] = rewards[i]
                    else:
                        targets_val[i] = rewards[i] + discountedtarget
                    error += torch.sum(abs(olds[i] - targets_val[i])).item()
                    if isinstance(error, np.ndarray):
                        error =  np.sum(error)
            self.memory.addorupdate(error, (items, actions, rewards, nextitems, nextmasks, done))

    # pick samples from prioritized replay memory (with batch_size)
    def train_model(self):
        self.model.train()
        self.target_model.train()
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            if self.epsilon<0:
                self.epsilon=0
        tryfetch = 0
        while tryfetch<3:
          mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
          #print("idxs, is_weights", len(idxs), len(is_weights))
          mini_batch = np.array(mini_batch, dtype=object).transpose()
          if any(not isinstance(arr, np.ndarray) for arr in mini_batch[0]) or any(not isinstance(arr, np.ndarray) for arr in mini_batch[2]):
              if tryfetch<3:
                  tryfetch += 1
              else:
                  print("mini_batch = ", mini_batch)
                  exit()
          else:
              break
        items = list(mini_batch[0]) 
        #print("items", items)
        max_lengthit = max(len(arr) for arr in items)
        #print("max_lengthit", max_lengthit)
        tempitems = np.zeros((len(items), max_lengthit, self.state_size))
        for i, sequence in enumerate(items):
            tempitems[i, :len(sequence), :] = sequence
        #tempitems = [np.pad(item, ((0, max_lengthit - len(item)))) for item in items]
        items = tempitems#np.vstack(tempitems)       
        #print("itemslen", len(items))
        actions = list(mini_batch[1])  
        #print("actions", actions)
        max_lengthac = max(arr.size for arr in actions)
        #print("max_lengthac", max_lengthac)
        hidenmasks = copy.deepcopy(actions)
        hidenmasks = [np.where(hidenmask == 0, 1, hidenmask) for hidenmask in hidenmasks]
        tempactions = []
        temphidenmasks = []
        tempactions = [np.pad(item, ((0, max_lengthac - item.size))) for item in actions]
        temphidenmasks = [np.pad(item, ((0, max_lengthac - item.size))) for item in hidenmasks]
        #print("tempactions", tempactions)
        actions = np.vstack(tempactions)
        #print("actions", actions)
        hidenmasks = np.vstack(temphidenmasks)
        rewards = list(mini_batch[2])
        max_lengthrw = max(arr.size for arr in rewards)
        temprewards = [np.pad(item, ((0, max_lengthrw - item.size))) for item in rewards]    
        rewards = np.vstack(temprewards)
        currentmasks = actions # not necessary becuase the action is used to mask pred below                                
        #print("rewards", len(rewards))                            
        nextitems = list(mini_batch[3]) 
        #print("nexitems", len(nextitems))
        nextmasks = list(mini_batch[4])                                
        dones = list(mini_batch[5])

        if max_lengthac!= max_lengthrw != max_lengthit:
            print("batchsize mismatch")
            #exit()   
                                
        # bool to binary
        dones = np.array(dones).astype(int)

        # Q function of current state
        states = torch.Tensor(items)
        states = Variable(states).float()
        #print("items", states.shape)
        actions = torch.FloatTensor(actions)
        hidenmasks = torch.FloatTensor(hidenmasks)
        pred = self.model(states, hidenmasks)
        #print("pred", pred.shape)
        #print("actions", actions.shape)
        #total loss of the seleced actions at every batch using the action as masking
        #pred = torch.sum(pred.mul(Variable(actions)), dim=1)
        #pred = torch.sum(pred.mul(actions.unsqueeze(-1)), dim=1)   
        pred = pred.mul(actions.unsqueeze(-1)) 
        #print("pred", pred.shape)
        # Q function of next state
        next_pred = torch.zeros(self.batch_size, 1)#The size of next states mast match the current states
        for n in range(self.batch_size): 
            if  len(nextitems[n])>0:
                inext_states = torch.Tensor(nextitems[n])
                inext_states = Variable(inext_states).float().unsqueeze(0)   
                inextmasks = nextmasks[n]
                inextmasks = torch.FloatTensor(inextmasks)
                next_pred[n] = torch.max((self.target_model(inext_states, None).data)*(inextmasks.reshape(len(nextitems[n]),1)))
                '''
                model_output = self.target_model(inext_states)
                print("nextitems[n]", nextitems[n], "model_output", model_output, "inextmasks", inextmasks)
                mask = torch.zeros((1, nextitems[n].shape[0], *inext_states.shape[1:]), dtype=torch.bool)
                mask[:, :, nextitems[n][:,0].long(), nextitems[n][:,1].long(), nextitems[n][:,2].long(), nextitems[n][:,3].long()] = True
                mask &= inextmasks.reshape(1, -1, 1)
                
                # use masked_select to get the unmasked elements of the model output tensor
                unmasked_output = torch.masked_select(model_output, mask)
                
                # find the maximum value of the unmasked elements
                next_pred[n] = torch.max(unmasked_output)
                '''
        #print("next_pred", next_pred)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        # Q Learning: get maximum Q value at s' from target model
        #print("actions", actions)
        #print("rewards", rewards)
        rewards = rewards.mul(Variable(actions))  # becuase we don't need reward and nexte states for the unselected  ones 
        dones = dones.reshape(self.batch_size, 1)
        #print("rewards", rewards.shape)
        #print("dones", dones)
        discounted_next_pred = (1 - dones) * self.discount_factor * next_pred
        #print("discountednextpred", discountednextpred)
        #print("shepeddiscountednextpred", actions * discounted_next_pred)
        target = rewards + actions * discounted_next_pred# masking the next pred with the actions, reward is already masked
        target = Variable(target)
        #print("target", target.shape)
        #target = torch.sum(target.mul(Variable(actions)), dim=1)  # becuase we don't need reward and nexte states for the unselected  ones                   
        #print("actions", actions)
        errors = torch.abs(pred.squeeze(dim=-1) - target).data.numpy()

        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            error = errors[i]
            if isinstance(error, np.ndarray):
                error =  np.sum(error)
            #print("errors",idx,errors[i])
            self.memory.update(idx, error)

        self.optimizer.zero_grad()

        # MSE Loss function
        loss = (torch.FloatTensor(is_weights) * F.mse_loss(pred.squeeze(dim=-1), target)).mean()
        #loss =  F.mse_loss(pred, target)
        loss.backward()

        # and train
        self.optimizer.step()
                                    
  
              

                
                
