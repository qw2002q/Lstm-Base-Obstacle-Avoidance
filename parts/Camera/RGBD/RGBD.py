###
#   Create disparity from Binocular Picture 
#   reference: https://www.cnblogs.com/zhiyishou/p/5767592.html
###

import cv2
import numpy as np
import os
# import camera_configs

class Camera_configs:
    def __init__(self):
        self.left_camera_matrix = np.array([[491.50710, 0., 492.13749],
                                       [0., 350.01779, 232.78032],
                                       [0., 0., 1.]])
        self.left_distortion = np.array([[0.11444, -0.06460, 0.00081, 0.00125, 0.00000]])
        
        
        
        self.right_camera_matrix = np.array([[497.04236, 0., 497.69553],
                                        [0., 262.72804, 225.82929],
                                        [0., 0., 1.]])
        self.right_distortion = np.array([[0.10404, -0.03600, -0.00102, -0.00015, 0.00000]])
        
        self.om = np.array([-0.00121, -0.00036, -0.00066])
        self.R = cv2.Rodrigues(self.om)[0]
        self.T = np.array([-52.29878, 0.10525, 4.53789])

camera_configs = Camera_configs()

class RGBD:
    def __init__(self, width = 640, height = 480):
        self.left = None
        self.right = None
        self.depth = None
        self.size = (width, height)

        R1, R2, P1, P2, self.Q, validPixROI1, validPixROI2 = cv2.stereoRectify(camera_configs.left_camera_matrix, camera_configs.left_distortion,
                                                                  camera_configs.right_camera_matrix, camera_configs.right_distortion, self.size, camera_configs.R,
                                                                  camera_configs.T)

        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(camera_configs.left_camera_matrix, camera_configs.left_distortion, R1, P1, self.size, cv2.CV_16SC2)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(camera_configs.right_camera_matrix, camera_configs.right_distortion, R2, P2, self.size, cv2.CV_16SC2)

    def divide(self, frame):
        self.left = frame[:, :self.size[0], :]
        self.right = frame[:, self.size[0]:, :]

        return self.left, self.right

    def save_left(self, seq = 0):
        RGBD_Path = "./RGBDs/"
        if not os.path.exists(RGBD_Path):
            os.makedirs(RGBD_Path)

        save_path = RGBD_Path + 'Left_' + str(seq) + '.jpg'
        cv2.imwrite(save_path, self.left)

    def save_right(self, seq = 0):
        RGBD_Path = "./RGBDs/"
        if not os.path.exists(RGBD_Path):
            os.makedirs(RGBD_Path)

        save_path = RGBD_Path + 'Right_' + str(seq) + '.jpg'
        cv2.imwrite(save_path, self.right)
    
    def create_RGBD(self, frame, num = 8, blockSize = 5):
        size = (frame.shape[1]/2, frame.shape[0])
        
        self.left = frame[:, :frame.shape[1]//2, :]
        self.right = frame[:, frame.shape[1]//2:, :]
        
        frame1 = self.left
        frame2 = self.right
        
        # Reconstruct the image according to the corrected map 
        img1_rectified = cv2.remap(frame1, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(frame2, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        
        # Set the picture as a grayscale image to prepare for stereo BM
        imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

        if blockSize % 2 == 0:
            blockSize += 1
        if blockSize < 5:
            blockSize = 5

        # num = 8 better in size(640, 480)
        # num = 5 better in size(320, 240)
        # Generate difference graph according to block machining method
        stereo = cv2.StereoBM_create(numDisparities=16*num, blockSize=blockSize)
        disparity = stereo.compute(imgL, imgR)

        disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Expand the picture into 3D space, and the value in Z direction is the current distance
        threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., self.Q)
        
        return disp, threeD
        

if __name__ == '__main__':
    rgbd = RGBD()
    base_load = "./save_imgs/"

    seq = 0
    seq_record = str(seq)
    # while (len(seq_record) < 6):
    #     seq_record = '0' + seq_record
    photos_path = base_load + 'test_' + seq_record + '.jpg'

    while os.path.exists(photos_path):
        imgs = cv2.imread(photos_path)
        rgbd.divide(imgs)
        rgbd.save_left(seq)
        rgbd.save_right(seq)
        rgbd.create_RGBD(imgs)

        seq += 1
        seq_record = str(seq)
        # while (len(seq_record) < 6):
        #     seq_record = '0' + seq_record
        photos_path = base_load + 'test_' + seq_record + '.jpg'

    print('found seq photos: ' + str(seq - 1))
