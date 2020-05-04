# filename: camera_configs.py
import cv2
import numpy as np

left_camera_matrix = np.array([[491.50710, 0., 492.13749],
                               [0., 350.01779, 232.78032],
                               [0., 0., 1.]])
left_distortion = np.array([[0.11444, -0.06460, 0.00081, 0.00125, 0.00000]])



right_camera_matrix = np.array([[497.04236, 0., 497.69553],
                                [0., 262.72804, 225.82929],
                                [0., 0., 1.]])
right_distortion = np.array([[0.10404, -0.03600, -0.00102, -0.00015, 0.00000]])

om = np.array([-0.00121, -0.00036, -0.00066])
R = cv2.Rodrigues(om)[0]
T = np.array([-52.29878, 0.10525, 4.53789])