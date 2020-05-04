###
#   Get Real Speed and Angle of the Vehicle
###

class Odom:
    def __init__(self):
        self.v = 0
        self.w = 0

    def callback(self, msg):
        self.v = float(int(msg.twist.twist.linear.x * 100)) / 100
        self.w = float(int(msg.twist.twist.angular.z * 100)) / 100

    def printData(self):
        print('speed = %s, angle = %s' % (self.v, self.w))




