"""
Getting started with XGBoost
============================

This is a simple example of using the native XGBoost interface, there are other
interfaces in the Python package like scikit-learn interface and Dask interface.


See :doc:`/python/python_intro` and :doc:`/tutorials/index` for other references.

"""
import numpy as np
import pickle
import xgboost as xgb
import os

from sklearn.datasets import load_svmlight_file

# Make sure the demo knows where to load the data.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XGBOOST_ROOT_DIR = CURRENT_DIR#os.path.dirname(os.path.dirname(CURRENT_DIR))
#DEMO_DIR = os.path.join(XGBOOST_ROOT_DIR, "demo")
DEMO_DIR = XGBOOST_ROOT_DIR


training = {}
training["binary_mnist"]  = "binary_mnist0"
training["mnist"]    = "ori_mnist.train0"
training["fashion"]  = "fashion.train0"
training["breast_cancer"] = "breast_cancer_scale0.train"
training["diabetes"] = "diabetes_scale0.train"
training["webspam"]  = "webspam_wc_normalized_unigram.svm0.train"
training["covtype"]  = "covtype.scale01.train0"
training["ijcnn"]    = "ijcnn1s0"
training["HIGGS"]    = "HIGGS_s.train0"


testing = {}
testing["binary_mnist"]  = "binary_mnist0.t"
testing["mnist"]    = "ori_mnist.test0"
testing["fashion"]  = "fashion.test0"
testing["breast_cancer"] = "breast_cancer_scale0.test"
testing["diabetes"] = "diabetes_scale0.test"
testing["webspam"]  = "webspam_wc_normalized_unigram.svm0.test"
testing["covtype"]  = "covtype.scale01.test0"
testing["ijcnn"]    = "ijcnn1s0.t"
testing["HIGGS"]    = "HIGGS_s.test0"


#dataset = "binary_mnist"
#dataset = "mnist"
#dataset = "fashion"
#dataset = "breast_cancer"
dataset = "diabetes"
#dataset = "webspam"
#dataset = "covtype"
#dataset = "ijcnn"
#dataset = "HIGGS"

if dataset == "binary_mnist":#------done-------done
	param = {"objective": "binary:logistic", "eta":0.02, "gamma":0.0, "min_child_weight":1, "max_depth": 4}
	num_round = 1000
elif dataset == "mnist":#------done-------done
	param = {"objective": "multi:softmax", "eta":0.3, "gamma":0.0, "min_child_weight":1, "max_depth": 8, "num_class": 10}
	num_round = 200
elif dataset == "fashion":#-----done--------done
	param = {"objective": "multi:softmax", "eta":0.3, "gamma":0.0, "min_child_weight":1, "max_depth": 8, "num_class": 10}
	num_round = 200
elif dataset == "breast_cancer":#-----done--------done
	param = {"objective": "binary:logistic", "eta":0.3, "gamma":1.0, "min_child_weight":1, "max_depth": 6}
	num_round = 10
elif dataset == "diabetes":#------done-------done
	param = {"objective": "binary:logistic", "eta":0.3, "gamma":1.0, "min_child_weight":1, "max_depth": 5}
	num_round = 25
elif dataset == "webspam":#------done-------done
	param = {"objective": "binary:logistic", "eta":0.3, "gamma":1.0, "min_child_weight":1, "max_depth": 8}
	num_round = 100
elif dataset == "covtype":#------done-------done
	param = {"objective": "multi:softmax", "eta":0.2, "gamma":0.0, "min_child_weight":1, "max_depth": 8, "num_class": 7}
	num_round = 200
elif dataset == "ijcnn":#------done-------done
	param = {"objective": "binary:logistic", "eta":0.3, "gamma":1.0, "min_child_weight":1, "max_depth": 8}
	num_round = 100
elif dataset == "HIGGS":#-------d---------
	param = {"objective": "binary:logistic", "eta":0.2, "gamma":1.0, "min_child_weight":1, "max_depth": 8}
	num_round = 300


#X_test, y_test = load_svmlight_file(os.path.join(DEMO_DIR, "", "inverted/data", testing[dataset]))
X_test, y_test         = load_svmlight_file(os.path.join(DEMO_DIR, "", "inverted/data", testing[dataset]))
X_test_inv, y_test_inv = load_svmlight_file(os.path.join(DEMO_DIR, "", "inverted/data", 'inverted_'+testing[dataset]))
X_test_flipped, y_test_flipped         = load_svmlight_file(os.path.join(DEMO_DIR, "", "inverted/data", 'flipped_'+testing[dataset]))
X_test_inv_flipped, y_test_inv_flipped = load_svmlight_file(os.path.join(DEMO_DIR, "", "inverted/data", 'inverted_flipped_'+testing[dataset]))
dtest     = xgb.DMatrix(X_test, y_test)
dtest_inv = xgb.DMatrix(X_test_inv, y_test_inv)
dtest_flipped     = xgb.DMatrix(X_test_flipped, y_test_flipped)
dtest_inv_flipped = xgb.DMatrix(X_test_inv_flipped, y_test_inv_flipped)


model_orig = xgb.Booster({'nthread': 20})
model_orig.load_model('inverted/models/'+dataset+'_orig.model')
model_both = xgb.Booster({'nthread': 20})
model_both.load_model('inverted/models/'+dataset+'_both.model')


# ORIG prediction =========================================================
preds = model_orig.predict(dtest,num_round)
labels = dtest.get_label()
#print(preds.shape[0], len(preds))
if param["objective"] == "binary:logistic":
    #print("error=%f" % (
    #  sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i])
    #    / float(len(preds))))
    print("ORIG accuracy=%f" % (1-
      sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i])
        / float(len(preds))))
else:
    preds = np.argmax(preds, axis=1)
    labels = labels.astype(int)
    #print('ORIG', np.sum(preds == labels), '/', preds.shape[0])
    print('ORIG', np.sum(preds == labels) / preds.shape[0])


# BOTH prediction =====================================================================
preds_inv = model_both.predict(dtest_inv,num_round)
labels = dtest_inv.get_label()
if param["objective"] == "binary:logistic":
    #print("error=%f" % (
    #  sum(1 for i in range(len(preds_inv)) if int(preds_inv[i] > 0.5) != labels[i])
    #  / float(len(preds_inv))))
    print("BOTH accuracy=%f" % (1-
      sum(1 for i in range(len(preds_inv)) if int(preds_inv[i] > 0.5) != labels[i])
        / float(len(preds_inv))))
else:
    preds_inv = np.argmax(preds_inv, axis=1)
    labels = labels.astype(int)
    #print('BOTH', np.sum(preds_inv == labels), '/', preds_inv.shape[0])
    print('BOTH', np.sum(preds_inv == labels) / preds_inv.shape[0])


# SUBSET prediction ======================================================================
if dataset == 'fashion':

  model_subset = xgb.Booster({'nthread': 20})
  model_subset.load_model('inverted/models/'+dataset+'_subgroup.model')
  preds_subgroup_1 = model_subset.predict(dtest,num_round)
  preds_subgroup_2 = model_subset.predict(dtest_inv,num_round)
  preds_subgroup_3 = model_subset.predict(dtest_flipped,num_round)
  preds_subgroup_4 = model_subset.predict(dtest_inv_flipped,num_round)
  
  labels = dtest.get_label()
  
  preds_1 = np.argmax(preds_subgroup_1, axis=1)
  preds_2 = np.argmax(preds_subgroup_2, axis=1)
  preds_3 = np.argmax(preds_subgroup_3, axis=1)
  preds_4 = np.argmax(preds_subgroup_4, axis=1)
  
  labels = labels.astype(int)
  #print('SUBSET', np.sum(preds == labels), '/', preds.shape[0])
  #print('SUBSET', np.sum(preds_subgroup == labels) / preds.shape[0])
      
  count = 0
  for ii in range(labels.shape[0]):
      if (preds_1[ii] == preds_2[ii]) or (preds_1[ii] == preds_3[ii]) or (preds_1[ii] == preds_4[ii]) or (preds_2[ii] == preds_3[ii]) or (preds_2[ii] == preds_4[ii]) or (preds_3[ii] == preds_4[ii]):
      	 count = count + 1
      	 
  print(count/labels.shape[0])
      

