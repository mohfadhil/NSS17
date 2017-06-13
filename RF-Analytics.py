#Paper presented at NSS17

# import the necessary packages
from random import randint
from sklearn import  decomposition, ensemble,preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import SparseRandomProjection,GaussianRandomProjection
from sklearn.cross_validation import train_test_split
from sklearn import metrics,tree
from sklearn import linear_model
import numpy as np
import pandas as pd

def changeorder(l):
    tup = ()
    l = [int(s) for s in l]
    l.sort()
    for a in l:
        tup = tup + (str(a),)
    return list(tup)

def changeorderindex(r):
    r = [int(s) for s in r]
    r.sort()
    r=[str(s) for s in r]
    return r

rfaccuracies ={}
rfrecall ={}
rff1 ={}
rfprec ={}
rfauc ={}
DataSet = pd.read_csv("Datasets/dataset8sepedited.csv", sep=",")
DataSet['agedbytwts']=DataSet['user.statuses_count']/DataSet['AccountAge']
DataSet['wbexl']=DataSetä['NoofExLinks']*DataSet['NoofWords']
npArray = np.array(DataSet)
X_digits = npArray[:,1:].astype(float)
y_digits = npArray[:,0]
le = preprocessing.LabelEncoder()
y_digits  = le.fit_transform(y_digits)
n = X_digits.shape[1]

dims = [1,3,5,7,9,11,13,15,17,19,21,25,31,37,41]
projs = [2,4,5,8,10,16,24,32,44]
minsamplesplit = [2,10,20,30,50,75,150,300,500]

split = train_test_split(X_digits, y_digits, test_size = 0.3,	random_state = 6)
(trainData, testData, trainTarget, testTarget) = split


for minsiplit in minsamplesplit:
    for exi in range (0,20):
        rfaccuracies ={}
        rfrecall ={}
        rff1 ={}
        rfprec ={}
        rfauc ={}
        results=pd.DataFrame()
        resultsrf=pd.DataFrame()
        resultspca=pd.DataFrame()
        rr = pd.DataFrame()


        for x in projs:
            rfaccuracies[str(x)]={}
            rfrecall[str(x)]={}
            rff1[str(x)]={}
            rfprec[str(x)]={}
            rfauc[str(x)]={}
            for t in dims:
                randomforests = ensemble.RandomForestClassifier(max_features =n,n_estimators=t,random_state=exi,max_depth= x,min_samples_split=minsiplit,n_jobs=-1)
                randomforests.fit(trainData, trainTarget)
                rfaccuracies[str(x)][str(t)]=metrics.accuracy_score( testTarget,randomforests.predict(testData))
                rfrecall[str(x)][str(t)]=metrics.recall_score( testTarget,randomforests.predict(testData))
                rff1[str(x)][str(t)]=metrics.f1_score( testTarget,randomforests.predict(testData))
                rfprec[str(x)][str(t)]=metrics.precision_score( testTarget,randomforests.predict(testData))
                rfauc[str(x)][str(t)]=metrics.roc_auc_score( testTarget,randomforests.predict(testData))


        writer = pd.ExcelWriter('results/ms'+str(minsiplit)+'exp'+str(exi)+'rf.xlsx')
        rfacc=pd.DataFrame.from_dict(rfaccuracies, orient='columns', dtype=None)
        rfrec=pd.DataFrame.from_dict(rfrecall, orient='columns', dtype=None)
        rff1m=pd.DataFrame.from_dict(rff1, orient='columns', dtype=None)
        rfprec=pd.DataFrame.from_dict(rfprec, orient='columns', dtype=None)
        rfroc=pd.DataFrame.from_dict(rfauc, orient='columns', dtype=None)

        rfacc = rfacc[changeorder(rfacc.columns.tolist())]
        rfacc=rfacc.reindex(changeorderindex(rfacc.index.tolist()))

        rfrec = rfrec[changeorder(rfrec.columns.tolist())]
        rfrec=rfrec.reindex(changeorderindex(rfrec.index.tolist()))

        rff1m = rff1m[changeorder(rff1m.columns.tolist())]
        rff1m=rff1m.reindex(changeorderindex(rff1m.index.tolist()))

        rfprec = rfprec[changeorder(rfprec.columns.tolist())]
        rfprec=rfprec.reindex(changeorderindex(rfprec.index.tolist()))

        rfroc = rfroc[changeorder(rfroc.columns.tolist())]
        rfroc=rfroc.reindex(changeorderindex(rfroc.index.tolist()))

        rfacc.to_excel(writer,sheet_name='rfaccuracies')
        rfrec.to_excel(writer,sheet_name='rfrecall')
        rff1m.to_excel(writer,sheet_name='rff1')
        rfprec.to_excel(writer,sheet_name='rfprec')
        rfroc.to_excel(writer,sheet_name='rfauc')
        writer.save()
