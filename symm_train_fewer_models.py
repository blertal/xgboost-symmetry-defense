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


both = {}
both["binary_mnist"]  = "both_binary_mnist0"
both["mnist"]         = "both_ori_mnist.train0"
both["fashion"]       = "both_fashion.train0"
both["breast_cancer"] = "both_breast_cancer_scale0.train"
both["diabetes"]      = "both_diabetes_scale0.train"
both["webspam"]       = "both_webspam_wc_normalized_unigram.svm0.train"
both["covtype"]       = "both_covtype.scale01.train0"
both["ijcnn"]         = "both_ijcnn1s0"
both["HIGGS"]         = "both_HIGGS_s.train0"


subgroup = {}
subgroup["binary_mnist"] = "subgroup_binary_mnist0"
subgroup["mnist"]        = "subgroup_ori_mnist.train0"
subgroup["fashion"]      = "subgroup_fashion.train0"



dataset = "covtype"

#print(both[dataset])
#exit()

if dataset == "binary_mnist":
	param = {"objective": "binary:logistic", "eta":0.02, "gamma":0.0, "min_child_weight":1, "max_depth": 4}
	num_round = 1000
elif dataset == "mnist":
	param = {"objective": "multi:softmax", "eta":0.3, "gamma":0.0, "min_child_weight":1, "max_depth": 8, "num_class": 10}
	num_round = 200
elif dataset == "fashion":
	param = {"objective": "multi:softmax", "eta":0.3, "gamma":0.0, "min_child_weight":1, "max_depth": 8, "num_class": 10}
	num_round = 200
elif dataset == "breast_cancer":
	param = {"objective": "binary:logistic", "eta":0.3, "gamma":1.0, "min_child_weight":1, "max_depth": 6}
	num_round = 10
elif dataset == "diabetes":
	param = {"objective": "binary:logistic", "eta":0.3, "gamma":1.0, "min_child_weight":1, "max_depth": 5}
	num_round = 25
elif dataset == "webspam":
	param = {"objective": "binary:logistic", "eta":0.3, "gamma":1.0, "min_child_weight":1, "max_depth": 8}
	num_round = 100
elif dataset == "covtype":
	param = {"objective": "multi:softmax", "eta":0.2, "gamma":0.0, "min_child_weight":1, "max_depth": 8, "num_class": 7}
	num_round = 200
elif dataset == "ijcnn":
	param = {"objective": "binary:logistic", "eta":0.3, "gamma":1.0, "min_child_weight":1, "max_depth": 8}
	num_round = 100
elif dataset == "HIGGS":
	param = {"objective": "binary:logistic", "eta":0.2, "gamma":1.0, "min_child_weight":1, "max_depth": 8}
	num_round = 300



ORIG = True
INV  = False
BOTH = False
SUBGROUP = False
ORIG_FLIPPED = False

# Orig dataset
if ORIG:
  X, y           = load_svmlight_file(os.path.join(DEMO_DIR, "inverted/data", training[dataset]))
  X_test, y_test = load_svmlight_file(os.path.join(DEMO_DIR, "", "inverted/data", testing[dataset]))
  
  #print(X.shape, y.shape)
  choices = np.random.choice(400000, 300000, replace=False)
  #print(choices.shape)
  #print(choices[0])
  X = X[choices, :]
  y = y[choices]
  
  dtrain = xgb.DMatrix(X, y)
  dtest = xgb.DMatrix(X_test, y_test)

# Inverted dataset
if INV:
  X_inv, y_inv           = load_svmlight_file(os.path.join(DEMO_DIR, "inverted/data", "inverted_"+training[dataset]))
  X_test_inv, y_test_inv = load_svmlight_file(os.path.join(DEMO_DIR, "", "inverted/data", "inverted_"+testing[dataset]))
  dtrain_inv = xgb.DMatrix(X_inv, y_inv)
  dtest_inv = xgb.DMatrix(X_test_inv, y_test_inv)


# Both dataset
if BOTH:
  X, y           = load_svmlight_file(os.path.join(DEMO_DIR, "inverted/data", both[dataset]))
  X_test, y_test = load_svmlight_file(os.path.join(DEMO_DIR, "", "inverted/data", testing[dataset]))
  dtrain = xgb.DMatrix(X, y)
  dtest = xgb.DMatrix(X_test, y_test)


if SUBGROUP:
  X, y           = load_svmlight_file(os.path.join(DEMO_DIR, "inverted/data", 'subgroup_' + training[dataset]))
  X_test, y_test = load_svmlight_file(os.path.join(DEMO_DIR, "inverted/data", testing[dataset]))
  dtrain = xgb.DMatrix(X, y)
  dtest = xgb.DMatrix(X_test, y_test)

  #print(X.shape, y.shape)
  #exit()

if ORIG_FLIPPED:
  X, y           = load_svmlight_file(os.path.join(DEMO_DIR, "inverted/data", 'orig_flipped_' + training[dataset]))
  X_test, y_test = load_svmlight_file(os.path.join(DEMO_DIR, "inverted/data", testing[dataset]))
  dtrain = xgb.DMatrix(X, y)
  dtest = xgb.DMatrix(X_test, y_test)


# Train
if ORIG:
  watchlist     = [(dtest, "eval"), (dtrain, "train")]
  bst = xgb.train(param, dtrain, num_boost_round=num_round, evals=watchlist)
  # run prediction
  preds = bst.predict(dtest,num_round)
  labels = dtest.get_label()
  if param["objective"] == "binary:logistic":
      print("error=%f" % (
        sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i])
        / float(len(preds))))
  else:
      preds = np.argmax(preds, axis=1)
      labels = labels.astype(int)
      print(np.sum(preds == labels), '/', preds.shape[0])


if INV:
  watchlist_inv = [(dtest_inv, "eval"), (dtrain_inv, "train")]
  bst_inv = xgb.train(param, dtrain_inv, num_boost_round=num_round, evals=watchlist_inv)
  # run prediction
  preds_inv = bst_inv.predict(dtest_inv,num_round)
  labels_inv = dtest_inv.get_label()
  if param["objective"] == "binary:logistic":
      print("error=%f" % (
        sum(1 for i in range(len(preds_inv)) if int(preds_inv[i] > 0.5) != labels_inv[i])
        / float(len(preds_inv))))
  else:
      preds_inv = np.argmax(preds_inv, axis=1)
      labels_inv = labels_inv.astype(int)
      print(np.sum(preds_inv == labels_inv), '/', preds_inv.shape[0])


# BOTH
if BOTH:
  watchlist     = [(dtest, "eval"), (dtrain, "train")]
  bst = xgb.train(param, dtrain, num_boost_round=num_round, evals=watchlist)
  # run prediction
  preds = bst.predict(dtest,num_round)
  labels = dtest.get_label()
  if param["objective"] == "binary:logistic":
      print("error=%f" % (
        sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i])
        / float(len(preds))))
  else:
      preds = np.argmax(preds, axis=1)
      labels = labels.astype(int)
      print(np.sum(preds == labels), '/', preds.shape[0])



# Train
if SUBGROUP:
  watchlist     = [(dtest, "eval"), (dtrain, "train")]
  bst = xgb.train(param, dtrain, num_boost_round=num_round, evals=watchlist)
  # run prediction
  preds = bst.predict(dtest,num_round)
  labels = dtest.get_label()
  if param["objective"] == "binary:logistic":
      print("error=%f" % (
        sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i])
        / float(len(preds))))
  else:
      preds = np.argmax(preds, axis=1)
      labels = labels.astype(int)
      print(np.sum(preds == labels), '/', preds.shape[0])


# Fashion orig and flipped training samples
if ORIG_FLIPPED:
  watchlist     = [(dtest, "eval"), (dtrain, "train")]
  bst = xgb.train(param, dtrain, num_boost_round=num_round, evals=watchlist)
  # run prediction
  preds = bst.predict(dtest,num_round)
  labels = dtest.get_label()
  if param["objective"] == "binary:logistic":
      print("error=%f" % (
        sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i])
        / float(len(preds))))
  else:
      preds = np.argmax(preds, axis=1)
      labels = labels.astype(int)
      print(np.sum(preds == labels), '/', preds.shape[0])


if ORIG:
    bst.dump_model('inverted/models/' + dataset + '_fewer.json', dump_format='json')
    bst.save_model('inverted/models/' + dataset + '_fewer.model')
    

if INV:
    bst_inv.dump_model('inverted/models/' + dataset + '_inv.json', dump_format='json')
    bst_inv.save_model('inverted/models/' + dataset + '_inv.model')


if BOTH:
    bst.dump_model('inverted/models/' + dataset + '_both.json', dump_format='json')
    bst.save_model('inverted/models/' + dataset + '_both.model')
    

if SUBGROUP:
    bst.dump_model('inverted/models/' + dataset + '_subgroup.json', dump_format='json')
    bst.save_model('inverted/models/' + dataset + '_subgroup.model')
    

if ORIG_FLIPPED:
    bst.dump_model('inverted/models/' + dataset + '_orig_flipped.json', dump_format='json')
    bst.save_model('inverted/models/' + dataset + '_orig_flipped.model')



