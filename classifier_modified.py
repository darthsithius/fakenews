import DataPrep
import FeatureSelection
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn import metrics
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

stopwords=set(STOPWORDS)
stopwords.update(["mr","mrs","ms","ad","al","say","says","will","said","take","dont","less","set","sat","ago","use","back",'aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 're', 'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn'])  

 
nb_pipeline = Pipeline([
        ('NBCV',FeatureSelection.countV),
        ('nb_clf',MultinomialNB())])


nb_pipeline.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_nb = nb_pipeline.predict(DataPrep.test_news['Statement'])
np.mean(predicted_nb == DataPrep.test_news['Label'])


logR_pipeline = Pipeline([
        ('LogRCV',FeatureSelection.countV),
        ('LogR_clf',LogisticRegression())
        ])

logR_pipeline.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_LogR = logR_pipeline.predict(DataPrep.test_news['Statement'])
np.mean(predicted_LogR == DataPrep.test_news['Label'])


svm_pipeline = Pipeline([
        ('svmCV',FeatureSelection.countV),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_svm = svm_pipeline.predict(DataPrep.test_news['Statement'])
np.mean(predicted_svm == DataPrep.test_news['Label'])


sgd_pipeline = Pipeline([
        ('svm2CV',FeatureSelection.countV),
        ('svm2_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5))
        ])

sgd_pipeline.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_sgd = sgd_pipeline.predict(DataPrep.test_news['Statement'])
np.mean(predicted_sgd == DataPrep.test_news['Label'])


random_forest = Pipeline([
        ('rfCV',FeatureSelection.countV),
        ('rf_clf',RandomForestClassifier(n_estimators=200,n_jobs=3))
        ])
    
random_forest.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_rf = random_forest.predict(DataPrep.test_news['Statement'])
np.mean(predicted_rf == DataPrep.test_news['Label'])


def build_confusion_matrix(classifier):
    
    k_fold = KFold(n_splits=10,shuffle=False)
    scores = []
    confusion = np.array([[0,0],[0,0]])

    for train_ind, test_ind in k_fold.split(DataPrep.train_news):
        train_text = DataPrep.train_news.iloc[train_ind]['Statement'] 
        train_y = DataPrep.train_news.iloc[train_ind]['Label']
    
        test_text = DataPrep.train_news.iloc[test_ind]['Statement']
        test_y = DataPrep.train_news.iloc[test_ind]['Label']
        
        classifier.fit(train_text,train_y)
        predictions = classifier.predict(test_text)
        
        confusion += confusion_matrix(test_y,predictions)
        score = f1_score(test_y,predictions)
        scores.append(score)
    
    return (print('Total statements classified:', len(DataPrep.train_news)),
    print('Score:', sum(scores)/len(scores)),
    print('score length', len(scores)),
    print('Confusion matrix:'),
    print(confusion))
print("line 118") 

print("Confusion matrix for naive")
build_confusion_matrix(nb_pipeline)
print("Confusion matrix for logR")
build_confusion_matrix(logR_pipeline)
print("Confusion matrix for svm")
build_confusion_matrix(svm_pipeline)
print("Confusion matrix for sgd")
build_confusion_matrix(sgd_pipeline)
print("Confusion matrix for foresr")
build_confusion_matrix(random_forest)
print("line 125")


nb_pipeline_ngram = Pipeline([
        ('nb_tfidf',FeatureSelection.tfidf_ngram),
        ('nb_clf',MultinomialNB())])

nb_pipeline_ngram.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_nb_ngram = nb_pipeline_ngram.predict(DataPrep.test_news['Statement'])
np.mean(predicted_nb_ngram == DataPrep.test_news['Label'])

logR_pipeline_ngram = Pipeline([
        ('LogR_tfidf',FeatureSelection.tfidf_ngram),
        ('LogR_clf',LogisticRegression(penalty="l2",C=1))
        ])

logR_pipeline_ngram.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_LogR_ngram = logR_pipeline_ngram.predict(DataPrep.test_news['Statement'])
np.mean(predicted_LogR_ngram == DataPrep.test_news['Label'])

svm_pipeline_ngram = Pipeline([
        ('svm_tfidf',FeatureSelection.tfidf_ngram),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline_ngram.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_svm_ngram = svm_pipeline_ngram.predict(DataPrep.test_news['Statement'])
np.mean(predicted_svm_ngram == DataPrep.test_news['Label'])

sgd_pipeline_ngram = Pipeline([
         ('sgd_tfidf',FeatureSelection.tfidf_ngram),
         ('sgd_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5))
         ])

sgd_pipeline_ngram.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_sgd_ngram = sgd_pipeline_ngram.predict(DataPrep.test_news['Statement'])
np.mean(predicted_sgd_ngram == DataPrep.test_news['Label'])

random_forest_ngram = Pipeline([
        ('rf_tfidf',FeatureSelection.tfidf_ngram),
        ('rf_clf',RandomForestClassifier(n_estimators=300,n_jobs=3))
        ])
    
random_forest_ngram.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_rf_ngram = random_forest_ngram.predict(DataPrep.test_news['Statement'])
np.mean(predicted_rf_ngram == DataPrep.test_news['Label'])
print("line 190")

build_confusion_matrix(nb_pipeline_ngram)
build_confusion_matrix(logR_pipeline_ngram)
build_confusion_matrix(svm_pipeline_ngram)
build_confusion_matrix(sgd_pipeline_ngram)
build_confusion_matrix(random_forest_ngram)


DataPrep.test_news['Label'].shape
print("line 208 ")

print("for TFIDF vectorizer")
print("f1,auc,accu score for rf")
print(metrics.f1_score(DataPrep.test_news['Label'], predicted_rf))
print(metrics.roc_auc_score(DataPrep.test_news['Label'], predicted_rf))
print(metrics.accuracy_score(DataPrep.test_news['Label'], predicted_rf))

print("f1,auc,accu score for logR")
print(metrics.f1_score(DataPrep.test_news['Label'], predicted_LogR))
print(metrics.roc_auc_score(DataPrep.test_news['Label'], predicted_LogR))
print(metrics.accuracy_score(DataPrep.test_news['Label'], predicted_LogR))

print("f1,auc,accu score for nb")
print(metrics.f1_score(DataPrep.test_news['Label'], predicted_nb))
print(metrics.roc_auc_score(DataPrep.test_news['Label'], predicted_nb))
print(metrics.accuracy_score(DataPrep.test_news['Label'], predicted_nb))

print("f1,auc,accu score for svm")
print(metrics.f1_score(DataPrep.test_news['Label'], predicted_svm))
print(metrics.roc_auc_score(DataPrep.test_news['Label'], predicted_svm))
print(metrics.accuracy_score(DataPrep.test_news['Label'], predicted_svm))

print("f1,auc,accu score for sgd")
print(metrics.f1_score(DataPrep.test_news['Label'], predicted_sgd))
print(metrics.roc_auc_score(DataPrep.test_news['Label'], predicted_sgd))
print(metrics.accuracy_score(DataPrep.test_news['Label'], predicted_sgd))

print("line 333")



#grid search
gs_clf = GridSearchCV(random_forest_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(DataPrep.train_news['Statement'][:10],DataPrep.train_news['Label'][:10])

print("line 275")
gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_
#logistic regression parameters
parameters = {'LogR_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
               'LogR_tfidf__use_idf': (True, False),
               'LogR_tfidf__smooth_idf': (True, False)
}
gs_clf = GridSearchCV(logR_pipeline_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(DataPrep.train_news['Statement'][:10],DataPrep.train_news['Label'][:10])
print("line 288")
gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_





print("for count vectorizer")
print("f1,auc,accu score for rf")
print(metrics.f1_score(DataPrep.test_news['Label'], predicted_rf_ngram))
print(metrics.roc_auc_score(DataPrep.test_news['Label'], predicted_rf_ngram))
print(metrics.accuracy_score(DataPrep.test_news['Label'], predicted_rf_ngram))

print("f1,auc,accu score for logR")
print(metrics.f1_score(DataPrep.test_news['Label'], predicted_LogR_ngram))
print(metrics.roc_auc_score(DataPrep.test_news['Label'], predicted_LogR_ngram))
print(metrics.accuracy_score(DataPrep.test_news['Label'], predicted_LogR_ngram))

print("f1,auc,accu score for nb")
print(metrics.f1_score(DataPrep.test_news['Label'], predicted_nb_ngram))
print(metrics.roc_auc_score(DataPrep.test_news['Label'], predicted_nb_ngram))
print(metrics.accuracy_score(DataPrep.test_news['Label'], predicted_nb_ngram))

print("f1,auc,accu score for svm")
print(metrics.f1_score(DataPrep.test_news['Label'], predicted_svm_ngram))
print(metrics.roc_auc_score(DataPrep.test_news['Label'], predicted_svm_ngram))
print(metrics.accuracy_score(DataPrep.test_news['Label'], predicted_svm_ngram))

print("f1,auc,accu score for sgd")
print(metrics.f1_score(DataPrep.test_news['Label'], predicted_sgd_ngram))
print(metrics.roc_auc_score(DataPrep.test_news['Label'], predicted_sgd_ngram))
print(metrics.accuracy_score(DataPrep.test_news['Label'], predicted_sgd_ngram))


if(metrics.f1_score(DataPrep.test_news['Label'], predicted_rf_ngram)>=metrics.f1_score(DataPrep.test_news['Label'], predicted_LogR_ngram)):
    abc=random_forest_ngram
else:
    abc=logR_pipeline_ngram

model_file = 'final_model.sav'
pickle.dump(abc,open(model_file,'wb'))
print("line 344")

def plot_learing_curve(pipeline,title):
    cv = KFold(n_splits=5,shuffle=True)
    
    X = DataPrep.train_news["Statement"]
    y = DataPrep.train_news["Label"]
    
    pl = pipeline
    pl.fit(X,y)
    
    train_sizes, train_scores, test_scores = learning_curve(pl, X, y, n_jobs=-1, cv=cv, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
       
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
     
    plt.figure()
    plt.title(title)
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()
    
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.ylim(-.1,1.1)
    plt.show()
    plt.savefig(title +'.png')
print("line 387")
print("for tfidf vect")
plot_learing_curve(logR_pipeline_ngram,"Naive-bayes Classifier(tfidf)")
plot_learing_curve(nb_pipeline_ngram,"LogisticRegression Classifier(tfidf)")
plot_learing_curve(svm_pipeline_ngram,"SVM Classifier(tfidf)")
plot_learing_curve(sgd_pipeline_ngram,"SGD Classifier(tfidf)")
plot_learing_curve(random_forest_ngram,"RandomForest Classifier(tfidf)")
print("line 394")
print("For cvect")


plot_learing_curve(nb_pipeline,"Naive-bayes_Classifier(cv)")
plot_learing_curve(logR_pipeline,"LogisticRegression_Classifier(cv)")
plot_learing_curve(svm_pipeline,"SVM_Classifier(cv)")
plot_learing_curve(sgd_pipeline,"SGD_Classifier(cv)")
plot_learing_curve(random_forest,"RandomForest_Classifier(cv)")

print("Finished")