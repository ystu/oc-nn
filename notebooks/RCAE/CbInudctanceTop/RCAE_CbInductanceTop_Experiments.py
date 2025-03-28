
PROJECT_DIR = "../../../"


import sys
sys.path.append(PROJECT_DIR)

from src.models.RCAE import RCAE_AD
import numpy as np 
from src.config import Configuration as Cfg
import time

DATASET = "CbInductanceTop"
IMG_HGT = 64 # 960  480  240  128 64 28
IMG_WDT= 64 # 1280  640  320  128 64 28
IMG_DIM = IMG_HGT * IMG_WDT
IMG_CHANNEL = 1
HIDDEN_LAYER_SIZE = 64 #32
MODEL_SAVE_PATH = PROJECT_DIR + "/models/" + DATASET + "/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/" + DATASET + "/RCAE/"
PRETRAINED_WT_PATH = ""
PredictMode = False # load the model and predict if true

# RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
RANDOM_SEED = [81]
AUC = []

for seed in RANDOM_SEED:
  startTime = time.time()
  Cfg.seed = seed
  rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH,seed,PROJECT_DIR)
  print("Train Data Shape: ",rcae.data._X_train.shape)
  print("Train Label Shape: ",rcae.data._y_train.shape)
  print("Validation Data Shape: ",rcae.data._X_val.shape)
  print("Validation Label Shape: ",rcae.data._y_val.shape)
  print("Test Data Shape: ",rcae.data._X_test.shape)
  print("Test Label Shape: ",rcae.data._y_test.shape)
  print("===========TRAINING AND PREDICTING WITH DCAE============================")
  if(PredictMode):
    auc_roc = rcae.predict_by_pretrain_model()
  else:
    auc_roc = rcae.fit_and_predict()

  AUC.append(auc_roc)
  print("time cost: %d seconds\n" %(time.time() - startTime))
  
# print("===========TRAINING AND PREDICTING WITH DCAE============================")
# print("AUROC computed ", AUC)
# auc_roc_mean = np.mean(np.asarray(AUC))
# auc_roc_std = np.std(np.asarray(AUC))
# print ("AUROC =====", auc_roc_mean ,"+/-",auc_roc_std)
# print("========================================================================")