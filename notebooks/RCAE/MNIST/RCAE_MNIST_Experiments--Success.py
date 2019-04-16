#!/usr/bin/env python
# coding: utf-8

# In[4]:


#from google.colab import drive
#drive.mount('/content/drive')
#get_ipython().system('pip install fuel')
#get_ipython().system('pip install picklable_itertools')

#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('reload_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


PROJECT_DIR = "/home/ubuntu-ai/anomaly_detection/oc-nn/"


import sys,os
import numpy as np
sys.path.append(PROJECT_DIR)


# # **RCAE-MNIST 9_Vs_all** 

# In[5]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from src.models.RCAE import RCAE_AD
import numpy as np 
from src.config import Configuration as Cfg

DATASET = "mnist"
IMG_DIM= 784
IMG_HGT =28
IMG_WDT=28
IMG_CHANNEL=1
HIDDEN_LAYER_SIZE= 32
MODEL_SAVE_PATH = PROJECT_DIR + "/models/MNIST/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/MNIST/RCAE/"
PRETRAINED_WT_PATH = ""
# RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
RANDOM_SEED = [42]
AUC = []

for seed in RANDOM_SEED:  
  Cfg.seed = seed
  rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH,seed)
  print("Train Data Shape: ",rcae.data._X_train.shape)
  print("Train Label Shape: ",rcae.data._y_train.shape)
  print("Validation Data Shape: ",rcae.data._X_val.shape)
  print("Validation Label Shape: ",rcae.data._y_val.shape)
  print("Test Data Shape: ",rcae.data._X_test.shape)
  print("Test Label Shape: ",rcae.data._y_test.shape)
  print("===========TRAINING AND PREDICTING WITH DCAE============================")
  auc_roc = rcae.fit_and_predict()
  print("========================================================================")
  AUC.append(auc_roc)
  
print("===========TRAINING AND PREDICTING WITH DCAE============================")
print("AUROC computed ", AUC)
auc_roc_mean = np.mean(np.asarray(AUC))
auc_roc_std = np.std(np.asarray(AUC))
print ("AUROC =====", auc_roc_mean ,"+/-",auc_roc_std)
print("========================================================================")


# # **RCAE-MNIST 7_Vs_all** 

# In[ ]:


# from src.models.RCAE import RCAE_AD
# import numpy as np 
# from src.config import Configuration as Cfg

# DATASET = "mnist"
# IMG_DIM= 784
# IMG_HGT =28
# IMG_WDT=28
# IMG_CHANNEL=1
# HIDDEN_LAYER_SIZE= 32
# MODEL_SAVE_PATH = PROJECT_DIR + "/models/MNIST/RCAE/"
# REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/MNIST/RCAE/"
# PRETRAINED_WT_PATH = ""
# RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
# AUC = []

# for seed in RANDOM_SEED:  
  # Cfg.seed = seed
  # rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH,seed)
  # print("Train Data Shape: ",rcae.data._X_train.shape)
  # print("Train Label Shape: ",rcae.data._y_train.shape)
  # print("Validation Data Shape: ",rcae.data._X_val.shape)
  # print("Validation Label Shape: ",rcae.data._y_val.shape)
  # print("Test Data Shape: ",rcae.data._X_test.shape)
  # print("Test Label Shape: ",rcae.data._y_test.shape)
  # print("===========TRAINING AND PREDICTING WITH DCAE============================")
  # auc_roc = rcae.fit_and_predict()
  # print("========================================================================")
  # AUC.append(auc_roc)
  
# print("===========TRAINING AND PREDICTING WITH DCAE============================")
# print("AUROC computed ", AUC)
# auc_roc_mean = np.mean(np.asarray(AUC))
# auc_roc_std = np.std(np.asarray(AUC))
# print ("AUROC =====", auc_roc_mean ,"+/-",auc_roc_std)
# print("========================================================================")


# **RCAE-MNIST 8_Vs_all**

# In[ ]:


# from src.models.RCAE import RCAE_AD
# import numpy as np 
# from src.config import Configuration as Cfg

# DATASET = "mnist"
# IMG_DIM= 784
# IMG_HGT =28
# IMG_WDT=28
# IMG_CHANNEL=1
# HIDDEN_LAYER_SIZE= 32
# MODEL_SAVE_PATH = PROJECT_DIR + "/models/MNIST/RCAE/"
# REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/MNIST/RCAE/"
# PRETRAINED_WT_PATH = ""
# RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
# AUC = []

# for seed in RANDOM_SEED:  
  # Cfg.seed = seed
  # rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH,seed)
  # print("Train Data Shape: ",rcae.data._X_train.shape)
  # print("Train Label Shape: ",rcae.data._y_train.shape)
  # print("Validation Data Shape: ",rcae.data._X_val.shape)
  # print("Validation Label Shape: ",rcae.data._y_val.shape)
  # print("Test Data Shape: ",rcae.data._X_test.shape)
  # print("Test Label Shape: ",rcae.data._y_test.shape)
  # print("===========TRAINING AND PREDICTING WITH DCAE============================")
  # auc_roc = rcae.fit_and_predict()
  # print("========================================================================")
  # AUC.append(auc_roc)
  
# print("===========TRAINING AND PREDICTING WITH DCAE============================")
# print("AUROC computed ", AUC)
# auc_roc_mean = np.mean(np.asarray(AUC))
# auc_roc_std = np.std(np.asarray(AUC))
# print ("AUROC =====", auc_roc_mean ,"+/-",auc_roc_std)
# print("========================================================================")


# **RCAE-MNIST 6_Vs_all**

# In[ ]:


# from src.models.RCAE import RCAE_AD
# import numpy as np 
# from src.config import Configuration as Cfg

# DATASET = "mnist"
# IMG_DIM= 784
# IMG_HGT =28
# IMG_WDT=28
# IMG_CHANNEL=1
# HIDDEN_LAYER_SIZE= 32
# MODEL_SAVE_PATH = PROJECT_DIR + "/models/MNIST/RCAE/"
# REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/MNIST/RCAE/"
# PRETRAINED_WT_PATH = ""
# RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
# AUC = []

# for seed in RANDOM_SEED:  
  # Cfg.seed = seed
  # rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH,seed)
  # print("Train Data Shape: ",rcae.data._X_train.shape)
  # print("Train Label Shape: ",rcae.data._y_train.shape)
  # print("Validation Data Shape: ",rcae.data._X_val.shape)
  # print("Validation Label Shape: ",rcae.data._y_val.shape)
  # print("Test Data Shape: ",rcae.data._X_test.shape)
  # print("Test Label Shape: ",rcae.data._y_test.shape)
  # print("===========TRAINING AND PREDICTING WITH DCAE============================")
  # auc_roc = rcae.fit_and_predict()
  # print("========================================================================")
  # AUC.append(auc_roc)
  
# print("===========TRAINING AND PREDICTING WITH DCAE============================")
# print("AUROC computed ", AUC)
# auc_roc_mean = np.mean(np.asarray(AUC))
# auc_roc_std = np.std(np.asarray(AUC))
# print ("AUROC =====", auc_roc_mean ,"+/-",auc_roc_std)
# print("========================================================================")


# **RCAE-MNIST 5_Vs_all**

# In[ ]:


# from src.models.RCAE import RCAE_AD
# import numpy as np 
# from src.config import Configuration as Cfg

# DATASET = "mnist"
# IMG_DIM= 784
# IMG_HGT =28
# IMG_WDT=28
# IMG_CHANNEL=1
# HIDDEN_LAYER_SIZE= 32
# MODEL_SAVE_PATH = PROJECT_DIR + "/models/MNIST/RCAE/"
# REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/MNIST/RCAE/"
# PRETRAINED_WT_PATH = ""
# RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
# AUC = []

# for seed in RANDOM_SEED:  
  # Cfg.seed = seed
  # rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH,seed)
  # print("Train Data Shape: ",rcae.data._X_train.shape)
  # print("Train Label Shape: ",rcae.data._y_train.shape)
  # print("Validation Data Shape: ",rcae.data._X_val.shape)
  # print("Validation Label Shape: ",rcae.data._y_val.shape)
  # print("Test Data Shape: ",rcae.data._X_test.shape)
  # print("Test Label Shape: ",rcae.data._y_test.shape)
  # print("===========TRAINING AND PREDICTING WITH DCAE============================")
  # auc_roc = rcae.fit_and_predict()
  # print("========================================================================")
  # AUC.append(auc_roc)
  
# print("===========TRAINING AND PREDICTING WITH DCAE============================")
# print("AUROC computed ", AUC)
# auc_roc_mean = np.mean(np.asarray(AUC))
# auc_roc_std = np.std(np.asarray(AUC))
# print ("AUROC =====", auc_roc_mean ,"+/-",auc_roc_std)
# print("========================================================================")


# **RCAE-MNIST 4_Vs_all**

# In[ ]:


# from src.models.RCAE import RCAE_AD
# import numpy as np 
# from src.config import Configuration as Cfg

# DATASET = "mnist"
# IMG_DIM= 784
# IMG_HGT =28
# IMG_WDT=28
# IMG_CHANNEL=1
# HIDDEN_LAYER_SIZE= 32
# MODEL_SAVE_PATH = PROJECT_DIR + "/models/MNIST/RCAE/"
# REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/MNIST/RCAE/"
# PRETRAINED_WT_PATH = ""
# RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
# AUC = []

# for seed in RANDOM_SEED:  
  # Cfg.seed = seed
  # rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH,seed)
  # print("Train Data Shape: ",rcae.data._X_train.shape)
  # print("Train Label Shape: ",rcae.data._y_train.shape)
  # print("Validation Data Shape: ",rcae.data._X_val.shape)
  # print("Validation Label Shape: ",rcae.data._y_val.shape)
  # print("Test Data Shape: ",rcae.data._X_test.shape)
  # print("Test Label Shape: ",rcae.data._y_test.shape)
  # print("===========TRAINING AND PREDICTING WITH DCAE============================")
  # auc_roc = rcae.fit_and_predict()
  # print("========================================================================")
  # AUC.append(auc_roc)
  
# print("===========TRAINING AND PREDICTING WITH DCAE============================")
# print("AUROC computed ", AUC)
# auc_roc_mean = np.mean(np.asarray(AUC))
# auc_roc_std = np.std(np.asarray(AUC))
# print ("AUROC =====", auc_roc_mean ,"+/-",auc_roc_std)
# print("========================================================================")


# **RCAE-MNIST 3_Vs_all**

# In[ ]:


# from src.models.RCAE import RCAE_AD
# import numpy as np 
# from src.config import Configuration as Cfg

# DATASET = "mnist"
# IMG_DIM= 784
# IMG_HGT =28
# IMG_WDT=28
# IMG_CHANNEL=1
# HIDDEN_LAYER_SIZE= 32
# MODEL_SAVE_PATH = PROJECT_DIR + "/models/MNIST/RCAE/"
# REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/MNIST/RCAE/"
# PRETRAINED_WT_PATH = ""
# RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
# AUC = []

# for seed in RANDOM_SEED:  
  # Cfg.seed = seed
  # rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH,seed)
  # print("Train Data Shape: ",rcae.data._X_train.shape)
  # print("Train Label Shape: ",rcae.data._y_train.shape)
  # print("Validation Data Shape: ",rcae.data._X_val.shape)
  # print("Validation Label Shape: ",rcae.data._y_val.shape)
  # print("Test Data Shape: ",rcae.data._X_test.shape)
  # print("Test Label Shape: ",rcae.data._y_test.shape)
  # print("===========TRAINING AND PREDICTING WITH DCAE============================")
  # auc_roc = rcae.fit_and_predict()
  # print("========================================================================")
  # AUC.append(auc_roc)
  
# print("===========TRAINING AND PREDICTING WITH DCAE============================")
# print("AUROC computed ", AUC)
# auc_roc_mean = np.mean(np.asarray(AUC))
# auc_roc_std = np.std(np.asarray(AUC))
# print ("AUROC =====", auc_roc_mean ,"+/-",auc_roc_std)
# print("========================================================================")


# **RCAE-MNIST 2_Vs_all**

# In[ ]:


# from src.models.RCAE import RCAE_AD
# import numpy as np 
# from src.config import Configuration as Cfg

# DATASET = "mnist"
# IMG_DIM= 784
# IMG_HGT =28
# IMG_WDT=28
# IMG_CHANNEL=1
# HIDDEN_LAYER_SIZE= 32
# MODEL_SAVE_PATH = PROJECT_DIR + "/models/MNIST/RCAE/"
# REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/MNIST/RCAE/"
# PRETRAINED_WT_PATH = ""
# RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
# AUC = []

# for seed in RANDOM_SEED:  
  # Cfg.seed = seed
  # rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH,seed)
  # print("Train Data Shape: ",rcae.data._X_train.shape)
  # print("Train Label Shape: ",rcae.data._y_train.shape)
  # print("Validation Data Shape: ",rcae.data._X_val.shape)
  # print("Validation Label Shape: ",rcae.data._y_val.shape)
  # print("Test Data Shape: ",rcae.data._X_test.shape)
  # print("Test Label Shape: ",rcae.data._y_test.shape)
  # print("===========TRAINING AND PREDICTING WITH DCAE============================")
  # auc_roc = rcae.fit_and_predict()
  # print("========================================================================")
  # AUC.append(auc_roc)
  
# print("===========TRAINING AND PREDICTING WITH DCAE============================")
# print("AUROC computed ", AUC)
# auc_roc_mean = np.mean(np.asarray(AUC))
# auc_roc_std = np.std(np.asarray(AUC))
# print ("AUROC =====", auc_roc_mean ,"+/-",auc_roc_std)
# print("========================================================================")


# **RCAE-MNIST 1_Vs_all**

# In[ ]:


# from src.models.RCAE import RCAE_AD
# import numpy as np 
# from src.config import Configuration as Cfg

# DATASET = "mnist"
# IMG_DIM= 784
# IMG_HGT =28
# IMG_WDT=28
# IMG_CHANNEL=1
# HIDDEN_LAYER_SIZE= 32
# MODEL_SAVE_PATH = PROJECT_DIR + "/models/MNIST/RCAE/"
# REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/MNIST/RCAE/"
# PRETRAINED_WT_PATH = ""
# RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
# AUC = []

# for seed in RANDOM_SEED:  
  # Cfg.seed = seed
  # rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH,seed)
  # print("Train Data Shape: ",rcae.data._X_train.shape)
  # print("Train Label Shape: ",rcae.data._y_train.shape)
  # print("Validation Data Shape: ",rcae.data._X_val.shape)
  # print("Validation Label Shape: ",rcae.data._y_val.shape)
  # print("Test Data Shape: ",rcae.data._X_test.shape)
  # print("Test Label Shape: ",rcae.data._y_test.shape)
  # print("===========TRAINING AND PREDICTING WITH DCAE============================")
  # auc_roc = rcae.fit_and_predict()
  # print("========================================================================")
  # AUC.append(auc_roc)
  
# print("===========TRAINING AND PREDICTING WITH DCAE============================")
# print("AUROC computed ", AUC)
# auc_roc_mean = np.mean(np.asarray(AUC))
# auc_roc_std = np.std(np.asarray(AUC))
# print ("AUROC =====", auc_roc_mean ,"+/-",auc_roc_std)
# print("========================================================================")


# **RCAE-MNIST 0_Vs_all**

# In[ ]:


# from src.models.RCAE import RCAE_AD
# import numpy as np 
# from src.config import Configuration as Cfg

# DATASET = "mnist"
# IMG_DIM= 784
# IMG_HGT =28
# IMG_WDT=28
# IMG_CHANNEL=1
# HIDDEN_LAYER_SIZE= 32
# MODEL_SAVE_PATH = PROJECT_DIR + "/models/MNIST/RCAE/"
# REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/MNIST/RCAE/"
# PRETRAINED_WT_PATH = ""
# RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
# AUC = []

# for seed in RANDOM_SEED:  
  # Cfg.seed = seed
  # rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH,seed)
  # print("Train Data Shape: ",rcae.data._X_train.shape)
  # print("Train Label Shape: ",rcae.data._y_train.shape)
  # print("Validation Data Shape: ",rcae.data._X_val.shape)
  # print("Validation Label Shape: ",rcae.data._y_val.shape)
  # print("Test Data Shape: ",rcae.data._X_test.shape)
  # print("Test Label Shape: ",rcae.data._y_test.shape)
  # print("===========TRAINING AND PREDICTING WITH DCAE============================")
  # auc_roc = rcae.fit_and_predict()
  # print("========================================================================")
  # AUC.append(auc_roc)
  
# print("===========TRAINING AND PREDICTING WITH DCAE============================")
# print("AUROC computed ", AUC)
# auc_roc_mean = np.mean(np.asarray(AUC))
# auc_roc_std = np.std(np.asarray(AUC))
# print ("AUROC =====", auc_roc_mean ,"+/-",auc_roc_std)
# print("========================================================================")


# Produce Embeddings

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Pretrain Autoencoder

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 
