###
#   Author: Seya Peng
#   To Record the Data
###

import os
import datetime
import cv2

class Recorder:
    def __init__(self, baseDataPath = 'DataRecord', curDataPath = 'Default'):
        self.baseDataPath = baseDataPath
        self.curDataPath = curDataPath
        self.recordInit = False
        self.seq = 0

        self.initRecord()

    def initRecord(self):
        self.recordInit = True
        self.seq = 0

        if not os.path.exists(self.baseDataPath):
            os.makedirs(self.baseDataPath)

        # get the currect time
        curr_time = datetime.datetime.now()
        if self.curDataPath == 'Default':
            self.curDataPath = str(curr_time.year) + '-' + str(curr_time.month) + '-' + str(curr_time.day) + '-' + str(
            curr_time.hour) + '-' + str(curr_time.minute) + '-' + str(curr_time.second)

    def recordData(self, speed, angle, photo = None):
        if not self.recordInit:
            self.initRecord()

        dataPath = self.baseDataPath + '/' + self.curDataPath
        if not os.path.exists(dataPath):
            os.makedirs(dataPath)
            os.makedirs(dataPath + '/data')
            os.makedirs(dataPath + '/photos')

        self.seq += 1
        seq_record = str(self.seq)
        while (len(seq_record) < 6):
            seq_record = '0' + seq_record

        # dataFile = dataPath + '/data/speedRecord_' + seq_record
        # file = open(dataFile, 'w')
        # file.write(str(speed) + ' ' + str(angle))
        # file.close()

        # record speed and angle
        dataFileAll = dataPath + '/speedRecord_ALL'
        file2 = open(dataFileAll, 'a')
        file2.write(str(speed) + " " + str(angle) + " " + str(seq_record) + '\n')
        file2.close()

        dataFileAll = dataPath + '/speedRecord_ALL_easyRead'
        file3 = open(dataFileAll, 'a')
        file3.write('speed: ' + str(speed) + ", angle: " + str(angle) + ', seq:' + str(seq_record) + '\n')
        file3.close()

        # record picture
        photoPath = dataPath + '/photos/photo_' + seq_record + '.jpg'
        cv2.imwrite(photoPath, photo)