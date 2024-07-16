import pickle
import numpy as np
from util import Evaluator

ev=Evaluator('./compactExtractionUV/benchmarkRecord/')

avgErrorList=[]
for i in range(33):
    with open('calibResults/%d.pkl'%(i+1),'rb') as f:
        calibDict=pickle.load(f)

    avgErrorList.append(ev.triangulateAndGetAvgError(calibDict))

e=np.array(avgErrorList)
print('Mean:',np.mean(e))
print('Std :',np.std(e))