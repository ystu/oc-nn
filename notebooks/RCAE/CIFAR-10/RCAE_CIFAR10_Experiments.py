#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')
get_ipython().system('pip install picklable_itertools')

get_ipython().system('pip install fuel')
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

PROJECT_DIR = "/content/drive/My Drive/2019/testing/oc-nn/"
import sys,os
import numpy as np
sys.path.append(PROJECT_DIR)


# ## Aeroplane Vs All 

# In[2]:



## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from src.models.RCAE import RCAE_AD
import numpy as np 
from src.config import Configuration as Cfg

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=3
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"

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


# ## Automobile Vs All 

# In[ ]:



## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from src.models.RCAE import RCAE_AD
import numpy as np 
from src.config import Configuration as Cfg

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=3
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"

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


# ## Bird Vs All 

# In[ ]:



## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from src.models.RCAE import RCAE_AD
import numpy as np 
from src.config import Configuration as Cfg

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=1
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"

PRETRAINED_WT_PATH = ""
RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
# RANDOM_SEED = [42]
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


# ## Cat Vs All 

# In[ ]:



## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from src.models.RCAE import RCAE_AD
import numpy as np 
from src.config import Configuration as Cfg

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=1
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"

PRETRAINED_WT_PATH = ""
RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
# RANDOM_SEED = [42]
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


# ## Deer Vs All 

# In[ ]:



## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from src.models.RCAE import RCAE_AD
import numpy as np 
from src.config import Configuration as Cfg

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=1
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"

PRETRAINED_WT_PATH = ""
RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
# RANDOM_SEED = [42]
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


# ## Dog Vs All 

# In[ ]:



## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from src.models.RCAE import RCAE_AD
import numpy as np 
from src.config import Configuration as Cfg

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=3
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"

PRETRAINED_WT_PATH = ""
RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
# RANDOM_SEED = [42]
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


# ## Frog Vs All

# In[ ]:



## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from src.models.RCAE import RCAE_AD
import numpy as np 
from src.config import Configuration as Cfg

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=3
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"

PRETRAINED_WT_PATH = ""
RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
# RANDOM_SEED = [42]
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


# ## Horse Vs All

# In[ ]:



## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from src.models.RCAE import RCAE_AD
import numpy as np 
from src.config import Configuration as Cfg

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=3
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"

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


# ## Truck Vs All 

# In[ ]:



## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from src.models.RCAE import RCAE_AD
import numpy as np 
from src.config import Configuration as Cfg

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=3
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"

PRETRAINED_WT_PATH = ""
RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
# RANDOM_SEED = [42]
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


# ##**SHIP vs All **##

# In[ ]:



## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from src.models.RCAE import RCAE_AD
import numpy as np 
from src.config import Configuration as Cfg

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=1
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"

PRETRAINED_WT_PATH = ""
RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
# RANDOM_SEED = [42]
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


# ##** Bird vs All **##

# In[ ]:



## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from src.models.RCAE import RCAE_AD
import numpy as np 
from src.config import Configuration as Cfg

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=3
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"

PRETRAINED_WT_PATH = ""
RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
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


# ##** Truck vs All **##

# In[ ]:


## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from src.models.RCAE import RCAE_AD

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=1
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"
PRETRAINED_WT_PATH = ""

rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH)

print("Train Data Shape: ",rcae.data._X_train.shape)
print("Train Label Shape: ",rcae.data._y_train.shape)
print("Validation Data Shape: ",rcae.data._X_val.shape)
print("Validation Label Shape: ",rcae.data._y_val.shape)
print("Test Data Shape: ",rcae.data._X_test.shape)
print("Test Label Shape: ",rcae.data._y_test.shape)
print("===========TRAINING AND PREDICTING WITH RCAE============================")
rcae.fit_and_predict()
print("========================================================================")


# ##** AIRPLANE vs All **##

# In[ ]:


## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from src.models.RCAE import RCAE_AD

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=1
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"
PRETRAINED_WT_PATH = ""

rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH)

print("Train Data Shape: ",rcae.data._X_train.shape)
print("Train Label Shape: ",rcae.data._y_train.shape)
print("Validation Data Shape: ",rcae.data._X_val.shape)
print("Validation Label Shape: ",rcae.data._y_val.shape)
print("Test Data Shape: ",rcae.data._X_test.shape)
print("Test Label Shape: ",rcae.data._y_test.shape)
print("===========TRAINING AND PREDICTING WITH RCAE============================")
rcae.fit_and_predict()
print("========================================================================")


# ##** Deer vs All **##

# In[ ]:


## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from src.models.RCAE import RCAE_AD

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=1
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"
PRETRAINED_WT_PATH = ""

rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH)

print("Train Data Shape: ",rcae.data._X_train.shape)
print("Train Label Shape: ",rcae.data._y_train.shape)
print("Validation Data Shape: ",rcae.data._X_val.shape)
print("Validation Label Shape: ",rcae.data._y_val.shape)
print("Test Data Shape: ",rcae.data._X_test.shape)
print("Test Label Shape: ",rcae.data._y_test.shape)
print("===========TRAINING AND PREDICTING WITH RCAE============================")
rcae.fit_and_predict()
print("========================================================================")


# ##** DOG vs All **##

# In[ ]:



## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from src.models.RCAE import RCAE_AD
import numpy as np 
from src.config import Configuration as Cfg

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=3
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"

PRETRAINED_WT_PATH = ""
RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
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


# ##** FROG vs All **##

# In[ ]:


## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from src.models.RCAE import RCAE_AD

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=1
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"
PRETRAINED_WT_PATH = ""

rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH)

print("Train Data Shape: ",rcae.data._X_train.shape)
print("Train Label Shape: ",rcae.data._y_train.shape)
print("Validation Data Shape: ",rcae.data._X_val.shape)
print("Validation Label Shape: ",rcae.data._y_val.shape)
print("Test Data Shape: ",rcae.data._X_test.shape)
print("Test Label Shape: ",rcae.data._y_test.shape)
print("===========TRAINING AND PREDICTING WITH RCAE============================")
rcae.fit_and_predict()
print("========================================================================")


# ##** AUTOMOBILE vs All **##

# In[ ]:


## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from src.models.RCAE import RCAE_AD

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=1
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"
PRETRAINED_WT_PATH = ""

rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH)

print("Train Data Shape: ",rcae.data._X_train.shape)
print("Train Label Shape: ",rcae.data._y_train.shape)
print("Validation Data Shape: ",rcae.data._X_val.shape)
print("Validation Label Shape: ",rcae.data._y_val.shape)
print("Test Data Shape: ",rcae.data._X_test.shape)
print("Test Label Shape: ",rcae.data._y_test.shape)
print("===========TRAINING AND PREDICTING WITH RCAE============================")
rcae.fit_and_predict()
print("========================================================================")


# ##** Cat vs All **##

# In[ ]:


## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from src.models.RCAE import RCAE_AD

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=1
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"
PRETRAINED_WT_PATH = ""

rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH)

print("Train Data Shape: ",rcae.data._X_train.shape)
print("Train Label Shape: ",rcae.data._y_train.shape)
print("Validation Data Shape: ",rcae.data._X_val.shape)
print("Validation Label Shape: ",rcae.data._y_val.shape)
print("Test Data Shape: ",rcae.data._X_test.shape)
print("Test Label Shape: ",rcae.data._y_test.shape)
print("===========TRAINING AND PREDICTING WITH RCAE============================")
rcae.fit_and_predict()
print("========================================================================")


# ##** HORSE vs All **##

# In[ ]:


## Obtaining the training and testing data
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from src.models.RCAE import RCAE_AD

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=1
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"
PRETRAINED_WT_PATH = ""

rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH)

print("Train Data Shape: ",rcae.data._X_train.shape)
print("Train Label Shape: ",rcae.data._y_train.shape)
print("Validation Data Shape: ",rcae.data._X_val.shape)
print("Validation Label Shape: ",rcae.data._y_val.shape)
print("Test Data Shape: ",rcae.data._X_test.shape)
print("Test Label Shape: ",rcae.data._y_test.shape)
print("===========TRAINING AND PREDICTING WITH RCAE============================")
rcae.fit_and_predict()
print("========================================================================")


# In[ ]:




