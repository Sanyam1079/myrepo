#importing necessary modules
import cv2
import numpy

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import UCF101
from torchvision.transforms import transforms
from pathlib import Path
import os
import shutil

#definiing the class customVideoDataset used to load dataset and segregate into labels and frames.
class customVideoDataset(Dataset): 
    def __init__(self, path, frame_count): #initialization of parameters needed to load the dataset ,i.e videos, labels and frames.
        self.videos = []
        self.labels = []
        self.frames = frame_count
        folder = Path(path)
        for label in sorted(os.listdir(folder)): #making a list of video/paths and labels
            for fname in os.listdir(os.path.join(folder, label)):
                self.videos.append(os.path.join(folder, label, fname))
                self.labels.append(label)

        self.label2index = {label: index for index, label in enumerate(sorted(set(self.labels)))}
        self.label_array = numpy.array([self.label2index[label] for label in self.labels], dtype=int)

    def __getitem__(self, idx):
        video = cv2.VideoCapture(self.videos[idx]) #using openCV to load/capture videos
        stacked_frames = numpy.empty(shape=(self.frames, 32, 32, 3),
                                     dtype=numpy.dtype('float16'))  # as frame would have shape h,w,channels
        frame_count = 0
        while video.isOpened() and frame_count<self.frames:   #frame count
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (32, 32))
            stacked_frames[frame_count] = frame
            frame_count += 1
        video.release()
        stacked_frames = stacked_frames.transpose((3, 0, 1, 2))

        return stacked_frames, self.label_array[idx]

    def __len__(self):
        length = len(self.videos) 
        return length

##loading the dataset using dataloader from torch.utils
def getDataloader(path, batch, workers, frames):
    dataset = customVideoDataset(path=path, frame_count=frames)
    dataloader = DataLoader(dataset, batch_size=batch, num_workers=workers, shuffle=True)
    return dataloader

##Run this once to get train test split of UFC101
def ufctraintest(root_dir, annotation_dir, target_dir):
    os.chdir(annotation_dir)
    files = os.listdir()
    train = []
    test = []
    for file in files:
        if 'train' in file:
            train.append(file)
        if 'test' in file:
            test.append(file)
    classes = open('classInd.txt', 'r')
    for Class in classes:
        try:
            os.chdir(target_dir + '/train')
            os.mkdir(Class.split()[1])
            os.chdir(target_dir + '/test')
            os.mkdir(Class.split()[1])
        except:
            pass

    classes.close()

    os.chdir(target_dir + '/train')
    classes = os.listdir()
    print(classes)
    print(len(classes))
    print('MOVING TRAINING FILES')

    for file in train:  ##generating training splits/their paths
        print(train)
        line = open(annotation_dir +'/'+file, 'r')
        for video in line:
            video = video.split('/')[1]
            for Class in classes:
                if Class in video:
                    try:
                        shutil.move(src=root_dir + '/' + video.split()[0], dst=target_dir + '/train/' + Class + '/')
                    except Exception as e:
                        print(e)
        line.close()


    os.chdir(target_dir + '/test')
    classes = os.listdir()
    print(classes)
    print(len(classes))
    print('MOVING TEST FILES')

    for file in test: ##generate test splits/their paths
        print(test)
        line = open(annotation_dir +'/'+file, 'r')
        for video in line:
            video =video.split('/')[1]
            for Class in classes:
                if Class in video:
                    print(video)
                    try:
                        shutil.move(src=root_dir + '/' + video, dst=target_dir + '/test/' + Class + '/')
                    except Exception as e:
                        print(e)
        line.close()
  ##train and test splits paths are genarated
