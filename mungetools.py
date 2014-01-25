import numpy as np
import pandas as pd
import csv
import re
'''
Data munging tools
'''

def loadData(trainfile='data/train.csv',testfile='data/test.csv'):
    tr=pd.read_csv(trainfile)
    test=pd.read_csv(testfile)
    return (tr,test)

def writeout(pred,testid,filename,header=["PassengerId","Survived"]):
    # save out to .csv
    f = open(filename,'wb')
    csvf=csv.writer(f)
    csvf.writerow(header)
    csvf.writerows(zip(testid,pred))
    f.close()

def binData(data,edges):
    '''
    Bin data into edges and return counts and indices
    edges are exclusive on bottom and inclusive on top
    returns counts which is len(edges)-1
    returns ndarray of len(data) with indices corresponding to bins
    data <= first edge is given -1 value and > last edge a -2 value
    '''
    idata=np.array(data)
    inds = np.zeros(len(data))
    bins=np.ndarray(len(edges)-1)
    # mask data below first bin
    mask=idata<=edges[0]
    inds[mask]=-1
    for i in range(len(edges)-1):
        mask = np.logical_and(idata>edges[i],idata<=edges[i+1])
        bins[i]=sum(mask)
        inds[mask]=i+1
    
    # data above last bin
    mask = idata>edges[-1]
    inds[mask]=-2
    return (bins,inds)

def catSplit(DF,key,uinds=[],memberthresh=0.03,otherlabel=0):
    '''
    Create new columns in data frame from groups
    Basically one-hot encoding
    new columns are binary depending on membership in group
    optional input of groups names, otherwise uses unique values
    optional threshold for at least memberthresh fraction in group
    optional group label for those not in groups specified
    note: changes the input dataframe in place!
    '''
    keydata = np.array(DF[key].fillna('NA'))
    if not(uinds):
        uinds = np.unique(keydata)     
    else:
        memberthresh = 0
    otherkey=np.zeros(len(keydata))
    otherkeyname = '%s_other' %key
    usedkeys = list()
    for i in range(len(uinds)):
        newkey = uinds[i]
        newkeyname = '%s_%s' %(key,newkey)
        mask = (keydata==newkey)
        if sum(mask)/float(len(mask))>=memberthresh:
            DF[newkeyname]=mask
            usedkeys.append(uinds[i])
        else:
            otherkey=otherkey+mask  
    if otherkey.any() and otherlabel:
        DF[otherkeyname]=otherkey
    DF=DF.drop(key,axis=1)
    return usedkeys

    
    
def getGroupStats(DF,statkey,lookupkey):
    '''
    Get average stats for column statkey for each group in lookupkey
    Assumes data that is binned in lookupkey
    returns a list with averages, counts, and subgroup name for each group
    '''
    statseries = np.array(DF[statkey])
    lookupseries = np.array(DF[lookupkey])
    uvals = np.unique(lookupseries)
    avgs = np.zeros(len(uvals))
    counts = np.zeros(len(uvals))
    for i in range(len(uvals)):
        subgroup = (statseries[lookupseries==uvals[i]])
        subgroup = subgroup[~np.isnan(subgroup)]
        avgs[i]= subgroup.mean()
        counts[i] = subgroup.size
        
    return list(zip(avgs,counts,uvals))
    
def mungeData(DF):
    # make binary male (1) and female (0)
    DF['Sex']=(DF.Sex=='male')
    
    # parse titles from names
    getTitle = lambda x: re.findall('[A-Za-z]*\.',x)[0][:-1]
    DF['Title'] = DF.Name.apply(getTitle)
    
    titles=['Mrs','Miss','Mr','Master']
    tused=catSplit(DF,'Title',titles)
    
    # ticket has fairly minimal use, but first character has some value
    DF['Ticket1']=DF.Ticket.apply(lambda x: x[0])
    ticketvals=['P','A','C','S','1','2','3','4','5','6','7','8','9']
    tickused=catSplit(DF,'Ticket1',ticketvals)
    # combine 4-9 as these high number tickets tended to not survive
    DF['T_hi']=(DF['Ticket1_4']+DF['Ticket1_5']+DF['Ticket1_6']+DF['Ticket1_7']
        +DF['Ticket1_8']+DF['Ticket1_9'])
    DF=DF.drop(['Ticket1_4','Ticket1_5','Ticket1_6','Ticket1_7','Ticket1_8',
        'Ticket1_9'],axis=1)
    
    # some predictive power in whether person has a family
    DF['hasfam']=(DF['Parch']+DF['SibSp'])>0
    # use a threshold age for child to avoid model overfitting with continuous data
    # usually overfits so not using here. Uncomment to try
    #DF['isChild']=DF.Age<=8
    
    # put Pclass on range 0 to 1 that way everything is on the same range
    DF.Pclass=(DF.Pclass-1)/2
    
    # cabin and embarcation are not very useful
    DF=DF.drop(['Name','Title','Fare','Cabin','Ticket','Ticket1','Parch','SibSp',
        'PassengerId','Age','Embarked'],axis=1)
    return DF

def addFamSurvivors(DF1,DF2):
    '''
    Add column to train(DF1) and test(DF2) data that indicates if it is known that there is a
    surviving family member
    '''
    getLN=lambda x: re.findall('[A-Za-z]*\,',x)[0][:-1]
    fn1=np.array((DF1.Name.apply(getLN) + np.char.mod('%d',DF1.SibSp + DF1.Parch + 1)))
    fn2=np.array((DF2.Name.apply(getLN) + np.char.mod('%d',DF2.SibSp + DF2.Parch + 1)))
    #DF1['Fam']=fn1
    #DF2['Fam']=fn2
    unames = np.unique(np.append(fn1,fn2))
    surv1=np.zeros(len(fn1))
    surv2=np.zeros(len(fn2))
    survivors = DF1.Survived
    # this is not very pythonic...
    for i in unames:
        #get inds
        ninds1 = fn1==i
        ninds2 = fn2==i
        if sum(ninds1)>1:
            s=survivors[ninds1]
            if sum(s)>0:
                #someone in the family is alive!
                surv2[ninds2]=1
                sinds=np.arange(ninds1.size)[ninds1.nonzero()]
                for j in sinds:
                    #but only mark true if not the only surviving member
                    n=ninds1.copy()
                    n[j]=0
                    if sum(survivors[n])>0:
                        surv1[j]=1
    DF1['FamSurvived']=surv1
    DF2['FamSurvived']=surv2
    return (DF1,DF2)
