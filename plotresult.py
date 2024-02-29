import numpy as np
from numpy import savetxt
from numpy import loadtxt
import matplotlib.pyplot as plt
import scipy.stats
def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data)
    #print("a=",a)
    N = len(a)
    meanr, se = np.mean(a, axis=0), scipy.stats.sem(a)
    #print("meanr=",meanr)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., N-1)
    return meanr, h
'''
plot performance
'''
Eps=50000 # max episodes
NumberOfRuns = 40
window = 1000
print("plotting for {} runs".format(NumberOfRuns))
firtindex = 0
TransformerDRLfirstIndex = firtindex # starting index of numebr of runs
# find minimum episode, because some runs may have run longer while some others run few eposodes
for i in range(NumberOfRuns):
    
    TransformerDRLscores = loadtxt('./TransformerDRLresults/rMLP'+str(TransformerDRLfirstIndex+i)+'savescores.csv', delimiter=',')
    TransformerDRLscores1q = loadtxt('./TransformerDRLresults/rMLP'+str(TransformerDRLfirstIndex+i)+'savescores1q.csv', delimiter=',')
    
    Eps=min(len(TransformerDRLscores),Eps)
    Eps=min(len(TransformerDRLscores1q),Eps)    
TransformerDRLallscores = np.zeros((NumberOfRuns,Eps))
TransformerDRLallscores1q = np.zeros((NumberOfRuns,Eps))
# now for ploting
for i in range(NumberOfRuns):
    TransformerDRLscores = loadtxt('./TransformerDRLresults/rMLP'+str(TransformerDRLfirstIndex+i)+'savescores.csv', delimiter=',')
    TransformerDRLscores1q = loadtxt('./TransformerDRLresults/rMLP'+str(TransformerDRLfirstIndex+i)+'savescores1q.csv', delimiter=',')  
    perActionDQNavscores, perActionDQNavscores1q, TransformerDRLavscores, TransformerDRLavscores1q = [],[],[],[]
    for j in range(Eps):
        maxj= max(0,j-window) #smothing the curve
        TransformerDRLavscores.append(np.mean(TransformerDRLscores[maxj:j+1]))
        TransformerDRLavscores1q.append(np.mean(TransformerDRLscores1q[maxj:j+1]))
    TransformerDRLallscores[i,:] = TransformerDRLavscores
    TransformerDRLallscores1q[i,:] = TransformerDRLavscores1q
TransformerDRLavall, TransformerDRLciall = mean_confidence_interval(TransformerDRLallscores, confidence=0.95)
peak_index = np.argmax(TransformerDRLavall)
print("Peak value is {} at {}".format(TransformerDRLavall[peak_index]*100, peak_index))
TransformerDRLavall1q, TransformerDRLciall1q = mean_confidence_interval(TransformerDRLallscores1q, confidence=0.95)
episodes = np.arange(len(TransformerDRLavall))


# plot the TransformerDRL
plt.plot(episodes, TransformerDRLavall*100, 'red', label="Coalition action selection")
plt.fill_between(episodes, (TransformerDRLavall*100-TransformerDRLciall*100), (TransformerDRLavall*100+TransformerDRLciall*100), color='red', alpha=.1)
plt.plot(episodes, TransformerDRLavall1q*100, 'purple', label="Sequential action selection")
plt.fill_between(episodes, (TransformerDRLavall1q*100-TransformerDRLciall1q*100), (TransformerDRLavall1q*100+TransformerDRLciall1q*100), color='purple', alpha=.1)

#labels, legends, and titles
plt.title('Comparing performance relative to offline optimal')
plt.xlabel('Episodes') 
plt.ylabel('Total reward in % of IP')
plt.legend(framealpha=1, frameon=True)
plt.savefig("./Performance_Coalition.png")


#=========================
# t-test
# Perform independent samples t-test
t_statistic, p_value = scipy.stats.ttest_ind(TransformerDRLavall, TransformerDRLavall1q)

# Output the results
print(f"t-statistic: {t_statistic}")
print(f"P-value: {p_value}")

# Interpret the results
alpha = 0.05  # significance level
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in mean rewards.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in mean rewards.")

#=========================


'''

plot complexity

'''

#Transformer
Eps = 50000
#first find the minimum episode using the following foorloop
for i in range(NumberOfRuns):
    TransformerDRLqcountsim = loadtxt('./TransformerDRLresults/rMLP'+str(TransformerDRLfirstIndex+i)+'saveqcounts.csv', delimiter=',')
    Eps=min(len(TransformerDRLqcountsim),Eps)
    TransformerDRLqcountseq = loadtxt('./TransformerDRLresults/rMLP'+str(TransformerDRLfirstIndex+i)+'saveqcounts1q.csv', delimiter=',')
    Eps=min(len(TransformerDRLqcountseq),Eps)
    TransformerDRLtimesim = loadtxt('./TransformerDRLresults/rMLP'+str(TransformerDRLfirstIndex+i)+'saveTimes.csv', delimiter=',')
    Eps=min(len(TransformerDRLtimesim),Eps)
    TransformerDRLtimeseq = loadtxt('./TransformerDRLresults/rMLP'+str(TransformerDRLfirstIndex+i)+'saveTimes1q.csv', delimiter=',')
    Eps=min(len(TransformerDRLtimeseq),Eps)
# initialize arrays for the runs    
allTransformerDRLqcountsim = np.zeros((NumberOfRuns,Eps))
allTransformerDRLqcountseq = np.zeros((NumberOfRuns,Eps))
allTransformerDRLtimesim = np.zeros((NumberOfRuns,Eps))
allTransformerDRLtimeseq = np.zeros((NumberOfRuns,Eps))

for i in range(NumberOfRuns):
    TransformerDRLqcountsim = loadtxt('./TransformerDRLresults/rMLP'+str(TransformerDRLfirstIndex+i)+'saveqcounts.csv', delimiter=',')
    TransformerDRLqcountseq = loadtxt('./TransformerDRLresults/rMLP'+str(TransformerDRLfirstIndex+i)+'saveqcounts1q.csv', delimiter=',')
    TransformerDRLtimesim = loadtxt('./TransformerDRLresults/rMLP'+str(TransformerDRLfirstIndex+i)+'saveTimes.csv', delimiter=',')
    TransformerDRLtimeseq = loadtxt('./TransformerDRLresults/rMLP'+str(TransformerDRLfirstIndex+i)+'saveTimes1q.csv', delimiter=',')

    allTransformerDRLqcountsim[i,:] = TransformerDRLqcountsim[0:Eps]
    allTransformerDRLqcountseq[i,:] = TransformerDRLqcountseq[0:Eps] 
    allTransformerDRLtimesim[i,:] = TransformerDRLtimesim[0:Eps]
    allTransformerDRLtimeseq[i,:] = TransformerDRLtimeseq[0:Eps]
      
avallTransformerDRLqcountsim = np.average(allTransformerDRLqcountsim, axis=0)
avallTransformerDRLqcountseq = np.average(allTransformerDRLqcountseq, axis=0)
avallTransformerDRLtimesim = np.average(allTransformerDRLtimesim, axis=0)
avallTransformerDRLtimeseq = np.average(allTransformerDRLtimeseq, axis=0)
# create the subplots
fig, axs = plt.subplots(1, 2, figsize=(6, 5))
fig.subplots_adjust(wspace=0.4)
# plot the first figure in the first subplot
i= 0

# plot the second figure in the second subplot
data = [avallTransformerDRLqcountsim, avallTransformerDRLqcountseq]
axs[i].boxplot(data, positions=[1, 1.3], showfliers=False)
axs[i].set_title('Number of executions')
axs[i].set_xlabel('A')
axs[i].set_ylabel('Number of executions per episode')
axs[i].set_xticks([1, 1.3])
axs[i].set_xticklabels(['1', '2'])
i+=1
# plot the third figure in the third subplot
data = [avallTransformerDRLtimesim, avallTransformerDRLtimeseq]
axs[i].boxplot(data, positions=[1, 1.3], showfliers=False)
axs[i].set_title('Running time')
axs[i].set_xlabel('B')
axs[i].set_ylabel('CPU time per episode in seconds')
axs[i].set_xticks([1, 1.3])
axs[i].set_xticklabels(['1', '2'])

# save the figure
plt.savefig("./Boxes_Coalition.png")

