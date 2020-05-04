#!/usr/bin/env python
import rospy
import os
from pynput.keyboard import Listener
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from parts.Subscriber.Subscriber import Odom
from parts.Controller.KeyboardController import KeyBoardController
from parts.Recorder.Recorder import Recorder
from parts.Camera.Camera import  Camera
from parts.ulits import *
from parts.Net.Model import *
from parts.Net.DataLoader import *

argvList = ArgvList(sys.argv)
load_path = argvList.getargv('load_path', 'None')
id = argvList.getargv('device', 'cpu')
device = torch.device(id)

if __name__ == '__main__':
    img_size = (320, 240)
    # subcriber init
    odom = Odom()
    FPS = 20    # save hz
    HZ = 60     # action hz
    __count_max = int(FPS / HZ)

    ### ros init ###
    rospy.init_node('ctrl_node',anonymous=False)
    pub = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
    # sub = rospy.Subscriber('/odometry/filtered',Odometry,odom_callback,queue_size=1)
    sub = rospy.Subscriber('/odometry/filtered', Odometry, odom.callback, queue_size=1)
    loop_rate = rospy.Rate(HZ)  # hz
    msg = Twist()
    # msg.linear.x = -0.05
    # msg.angular.z = 0.0

    ### controller init ###
    controller = KeyBoardController(0.1, 0.5, 0.5, 0.7) # speed, angle, max_speed, reverse_speed_rate
    ### Recorder init ###
    recorder = Recorder()
    ### Camera init ###
    camera = Camera(img_size[0], img_size[1], 0, Binocular = True)  # width, hegiht, device_id, FPS
    # create Net
    modelfactory = ModelFactory(3, 2, img_size[0], img_size[1])
    # load model
    if(not load_path == 'None'):
        load_path = os.path.dirname(os.path.abspath(__file__)) + '/models/' + load_path
        model = modelfactory.load(load_path)
        model = model.to(device)
        model.print_params()

        transforms_ = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.48, 0.48, 0.48), (0.22, 0.22, 0.22)),
        ])
        imgbuffer = ImgBuffer(transforms_, 1, img_size[0], img_size[1])
        if(model.name == 'lstm'):
            imgbuffer.seq_length = model.seq_length
        
    count = 0

    while not rospy.is_shutdown() and not controller.shutdown:
        controller.listener()

        if controller.use_model:
            if load_path == 'None':
                print('Error: Model Unloaded')
                controller.use_model = False
            else:
                imgbuffer.updateBuffer(camera.frame, num=6, blockSize = 5)
                img = imgbuffer.getBuffer()['image'].float().to(device)
                disp = imgbuffer.getBuffer()['disp'].float().to(device)

                output = model(img, disp)
                speed = output[0][0].item()
                angle = output[0][1].item()
        else:
            speed = controller.speed
            angle = controller.angle

        msg.linear.x = speed
        msg.angular.z = angle

        ### update information ###
        camera.showMedia()
        controller.render()
        pub.publish(msg)
        loop_rate.sleep()

        ### print speed and angle Data ###
        count -= 1
        if count <= 0:
            # odom.printData()
            if controller.record:
                recorder.recordData(odom.v, odom.w, camera.frame)    # record speed, angle, photo
            count = __count_max
        # os.system('clear')

    controller.destory()
    camera.destroy()

