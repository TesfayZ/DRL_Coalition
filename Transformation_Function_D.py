import numpy as np
import time
# update the distinguishing indexes for the duplicate states
def distingusher(inputs):
    tempstate = inputs
    inds = np.lexsort(np.fliplr(tempstate).T)# save the index for later use, becuase the states must be arranged in accordance with the tasks
    sortedstates = tempstate[inds]
    distinguishers = np.ones(len(sortedstates))# initialize the distinguishers to 1, 1 means it is unique state, it will be increamented for every similar state
    #distinguishers = distinguishers # reset them to 1, it is assuming all unique
    for i in range(len(sortedstates)):
        if i>0: # the fist state is distinguised by 0 as there is no similar state before it
            if (sortedstates[i,]==sortedstates[i-1,]).all():# a duplicate is found
                distinguishers[i]=distinguishers[i-1]+1 # this does not need while loop becuase it is not increament by one from itself but form the previous one, So, it finishes it in one loop
    #normalize the distiguishers to 0 upto 1
    #distinguishers = distinguishers # update the uniqueness
    #distinguishers =distinguishers/max(distinguishers)
    # update the states by the distinguishers
    sortedstates[:,0] = distinguishers
    #now restore them to their original order to match their tasks
    originalinds = np.argsort(inds)
    State = sortedstates[originalinds]  
    return State
