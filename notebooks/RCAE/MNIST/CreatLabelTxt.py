import  os

train_path = "/home/ubuntu-ai/anomaly_detection/oc-nn/data/CbInductanceTop/train_128x128/"
test_path = "/home/ubuntu-ai/anomaly_detection/oc-nn/data/CbInductanceTop/test_128x128/"

listTrain = os.listdir(train_path)
print(listTrain)

for name in listTrain:
    open(train_path + name.split(".")[0] + ".txt", "w+")

listTest = os.listdir(test_path)
print(listTest)

for name in listTest:
    f = open(test_path + name.split(".")[0] + ".txt", "w+")
    f.write("1")

