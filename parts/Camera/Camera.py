###
#   Author: Seya Peng
#   Use the Camera
#   Parameters: Binocular -> True to use the binocular camera
###

import cv2
import numpy
import os
from PIL import Image
## import Superior documents
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/RGBD')
from RGBD import *
sys.path.pop()  # resume after using

import time

class Camera:
    __Is_camera = False

    def __init__(self, width = -1, height = -1, device = 0, FPS = 60, Binocular = False):
        if self.__Is_camera_inited():
            print('** Error: Can not create two camera before destory another! **')
            exit()

        if width == -1:
            width = 640
        if height == -1:
            height = 480 * (width / 640)

        if Binocular:
            width = width * 2

        self.width = width
        self.height = height

        self.__Change_init_state()
        self.camera = cv2.VideoCapture(device)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)   # set width
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)  # set height
        self.camera.set(cv2.CAP_PROP_FPS, FPS)    # set FPS
        
        self.ret, self.frame = self.camera.read()
        print("Camera resolution: " + str(self.frame.shape))
    
    def showMedia(self, update = True):
        # get a frame
        if update:
            self.ret, self.frame = self.camera.read()
        # show a frame
        cv2.imshow("capture", self.frame)
        return cv2.waitKey(delay = 1)

    def update(self):
        self.ret, self.frame = self.camera.read()
        return self.ret, self.frame

    def destroy(self):
        cv2.destroyAllWindows()
        self.camera.release()
        self.__Change_init_state()
        
    def save(self, seq = 0):
        print('save ' + str(seq))
        self.ret, self.frame = self.camera.read()
        print(self.frame.shape)
        fps = self.camera.get(cv2.CAP_PROP_FPS)
        print(fps)
        if not os.path.exists('./save_imgs'):
            os.makedirs('./save_imgs')
        place = './save_imgs/test_' + str(seq) + '.jpg'
        cv2.imwrite(place, self.frame)

    @classmethod
    def __Is_camera_inited(cls):
        return cls.__Is_camera

    @classmethod
    def __Change_init_state(cls):
        cls.__Is_canera = cls.__Is_camera  ^ True

if __name__ == '__main__':
    size = (640,480)
    camera = Camera(size[0], size[1], 1, Binocular = True)    # width, hegiht, device_id, FPS
    rgbd = RGBD(size[0], size[1])
    save_media = False
    count = 0

    time_count = 0
    t1 = time.time()

    cv2.namedWindow("disp")
    cv2.createTrackbar("num", "disp", 0, 10, lambda x: None)
    cv2.createTrackbar("blockSize", "disp", 5, 255, lambda x: None)
    while(1):
        time_count += 1
        if time_count % 100 == 99:
            print("FPS: %d" %(time_count / (time.time() - t1)))
            time_count = 0
            t1 = time.time()

        _, frame = camera.update()

        num = cv2.getTrackbarPos("num", "disp")
        blockSize = cv2.getTrackbarPos("blockSize", "disp")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        disp, _ = rgbd.create_RGBD(frame, num, blockSize)

        cv2.imshow("disp", disp)
        key = camera.showMedia(update = False)

        if key == ord('r'):
            save_media = save_media ^ True
            
        # save media
        if save_media:
            camera.save(count)
            count += 1
            
        if key == ord('q'):
            break

        if key == ord('s'):
            camera.save(count)
            count += 1
    camera.destroy()