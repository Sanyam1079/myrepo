#importing the codes from other .py files
from Model import spatioTemporalClassifier
from Dataloaders import getDataloader, ufctraintest
#main fucntion calls to train the model 
#if frame counts are different then batch size can only be 1
data = getDataloader('/kaggle/input/ucf101/UCF101/UCF-101', batch=2, workers=6, frames=102)
model = spatioTemporalClassifier(classes=2)
model.train_model(model=model, dataloader=data, epochs=2)
