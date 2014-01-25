import numpy as np
import mungetools as mg
from sklearn.svm import SVC

'''
Use Support Vector Machine classifier to predict Titanic survivors
Uses training data in train.csv
predicts from test.csv
As is this gets about 77.5% accuracy on the test set
'''

def trainClassifier(DF,paramc,paramg,split=0.5):
    '''
    Explore parameter values for SVM by splitting dataset in train and cv set
    Train each SVM classifier with parameter value and test on cv set
    returns best model
    '''
    nvals = len(DF)
    splitind=np.floor(nvals*split)
    nparams = len(paramc)*len(paramg)
    scores = np.zeros(nparams)
    counter=0
    paramholder=np.zeros([nparams,2])
    # randomize trainingset and split to train and CV set
    rp = np.random.permutation(nvals)
    survt=np.array(DF.iloc[rp[0:splitind],0])
    survcv=np.array(DF.iloc[rp[splitind:nvals],0])
    # note: SVM takes -1 and 1 for single class labels
    survt[~survt]=-1
    survcv[~survcv]=-1
    tset = DF.iloc[rp[0:splitind],1:]
    cvset = DF.iloc[rp[splitind:nvals],1:]
    bestscore=-1
    # loop through the values of c and g
    for c in paramc:
        for g in paramg:
            model=SVC(C=c,gamma=g)
            model=model.fit(tset,survt)
            try:
                scorei=model.score(cvset,survcv)
            except:
                scorei=0 # something went wrong...
            scores[counter]=np.mean(scorei)
            paramholder[counter,0]=c
            paramholder[counter,1]=g
            if scorei>bestscore:
                bestscore=scorei
                bestmodel=model
            counter+=1
            print('Score = %f with c: %f, g: %f' %(scorei,c,g))
    bestc=paramholder[scores.argmax(),0]
    bestg=paramholder[scores.argmax(),1]
    print('Best score of %f with c: %f, g: %f' %(bestscore,bestc,bestg))
    return bestmodel

    
# load in and munge data:

trdata,testdata=mg.loadData()

testid = np.array(testdata.PassengerId)

trdata,tesrdata=mg.addFamSurvivors(trdata,testdata)

trdata=mg.mungeData(trdata)
testdata=mg.mungeData(testdata)

# initialize classifier
# try several values of c (prediction error weight) and g (kernel width)
testc = [0.05, 0.1, 0.3, 0.6, 1, 3, 5, 10 ]
testg = [0, 0.01, 0.05, 0.1, 0.5, 1, 1.5]

# find the best model
model= trainClassifier(trdata,testc,testg)

# generate predictions
preds = model.predict(testdata)
preds=(preds>0)*1 # predictions are -1 and 1, so make 0 and 1
mg.writeout(preds,testid,'predictions/svmmodel_test.csv')
