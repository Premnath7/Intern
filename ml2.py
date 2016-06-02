import json
from sklearn import ensemble
import numpy as np
import pandas as pd
import csv
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
xs=np.zeros((44873,322), dtype=np.int)
ys=np.zeros((44873,), dtype=np.int)
#ys=[]
#xs=[]
scores_gbm=[]
j=0
with open('features','rb') as a:
	r = csv.reader(a)
	#print(type(r))
	for row in r:
		#print(len(row))
		if(row[1677]=='0' or row[1677]=='1'):
			#xs.append([])
			m=0
			for i in range(len(row)):
				if(i<300 or(i>1654 and i<len(row)-1)):
					#xs[len(xs)-1].append(int(row[i]))
					xs[j][m]=int(row[i])
					m=m+1
				if(i==len(row)-1):
					ys[j]=int(row[i])
		j=j+1
"""
for event in eh:
	if(event[len(event)-1]==0 or event[len(event)-1]==1):
		xs.append([])
		for i in range(len(event)):
			if(i<len(event)-1):
				xs[len(xs)-1].append(event[i])
			if(i==len(event)-1):
				ys.append(event[i])
"""
#xs = np.array(xs)
#ys = np.array(ys)
l=[100,200,300,400,500,600,700,800,900]
for x in l:
	s=0
	for train_index, test_index in cross_validation.StratifiedKFold(ys, n_folds=5):
				xs_train, xs_test = xs[train_index], xs[test_index]
				ys_train, ys_test = ys[train_index], ys[test_index]
				#xs_train = xs_train.toarray()
				#xs_test = xs_test.toarray()
				#xs_test=xs_train
				#ys_test=ys_train
				params = {'n_estimators': x, 'max_depth': 5, 'learning_rate': 0.1, 'min_samples_leaf': 50}	
				clf_gbm = ensemble.GradientBoostingClassifier(**params)
				clf_gbm.fit(xs_train, ys_train)
				ys_pred = clf_gbm.predict(xs_test)
				ys_pred_prob = clf_gbm.predict_proba(xs_test)			
				#scores_gbm.append(accuracy_score(ys_test, ys_pred)*100)
				s=s+accuracy_score(ys_test, ys_pred)*100
				"""
				print accuracy_score(ys_test, ys_pred)*100
				print classification_report(ys_test,ys_pred)
				print confusion_matrix(ys_test,ys_pred)
				d_test = {'y_pred_prob' : pd.Series(ys_pred_prob[:,1]), 'y' : pd.Series(ys_test), 'y_pred' : pd.Series(ys_pred)}
				df_test = pd.DataFrame(d_test).sort_values(by="y_pred_prob",ascending=False)
				df_test['cum_sum'] = df_test['y'].cumsum()
				df_test['cum_perc'] = 100*df_test.cum_sum/df_test['y'].sum()
				print df_test.loc[::int(ys_test.size/10),['cum_perc']].transpose().values
				print("-------------END OF CASE----------------------------")
				"""
	avg=s/5
	print(avg,x)