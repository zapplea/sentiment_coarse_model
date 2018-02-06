from af_unittest import AFTest
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7" ## 0

AF_model = AFTest()
classifier = AF_model.cls
config = AF_model.nn_config
print(config)
classifier.train()