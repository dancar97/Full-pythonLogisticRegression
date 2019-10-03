#Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


#Get the data
df = pd.read_csv('Your_data_here.csv',sep=';'  , engine='python')
df = shuffle(df)


feature_cols = ["Your Feature columns here (Independent variables)"]

y = df.identificador 


#---------------------------------------------------

#Select your categorical variables onto cat_vars to turn them into multiple coumns, each with a single category
cat_vars=[]
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(df[var], prefix=var)
    data1=df.join(cat_list)
    df=data1
cat_vars=[]
data_vars=df.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=df[to_keep]
#-----------------------------------------------


#Using RFE to generate a regression and get, support and ranking for each var
#If var = False, then it is not considered relevant to the model
#this is optional
X = data_final
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(X, y.values.ravel())

print(X.columns.values)
print(rfe.support_)
print(rfe.ranking_)
#.....................................




X = X.drop(['Here is your dependant variable, if it is on y, it should not be on X'], axis=1)



#train the model!
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)


#Get the conf matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)


#get various data and the matrix
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

print("CF matrix:")
print(cnf_matrix)


#confirma and generate simple table with SM
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

#Get new classification report to compare with prev results
print(classification_report(y_test, y_pred))



#get the ROC_AUC graph
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()