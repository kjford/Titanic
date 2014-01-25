import numpy as np
import mungetools as mg
from sklearn.ensemble import RandomForestClassifier as rfc

'''
Use random forrest classifier to predict Titanic survivors
Uses training data in train.csv (found in data subfolder)
predicts from test.csv
writes out to .csv in predictions subfolder
As is, this gives ~77% accuracy on test set
This can hit ~79% with some tweaking (currently overfits)
'''

# load data into pandas data frame
trdata,testdata=mg.loadData()

# get the id's for the test set
testid = np.array(testdata.PassengerId)

# determine if each passenger has a known surviving family member
trdata,tesrdata=mg.addFamSurvivors(trdata,testdata)

# munge the data to generate one-hot labels for gender, titles, ticket departments
trdata=mg.mungeData(trdata)
testdata=mg.mungeData(testdata)


# initialize classifier

model= rfc(n_estimators=1000,oob_score=True,compute_importances=True)

model = model.fit(trdata.iloc[:,1:],trdata.iloc[:,0])

accur = model.oob_score_

print('Out of Bag accuracy: %f \n' %accur)

# generate predictions
preds = model.predict(testdata)

# save out
mg.writeout(preds,testid,'predictions/rfcmodel_test.csv')