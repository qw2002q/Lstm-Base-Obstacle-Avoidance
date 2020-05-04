###
#   Author: Seya Peng
#   Controll the vehicle by KeyBoard
#   Direction Key: to controll moving
#   Shift: to speed up
#   Ctrl: to slow down
#   R: to start record
#   ESC: to Quit
###
import pygame
import sys
import time

class KeyBoardController:
    __Is_pygameinit = False

    @classmethod
    def __init__(self, v = 0.05, a = 0.05, max_v = 0.2, rv_rate = 0.6):
        if not self.Is_init():
            pygame.init()
            self.Inited()
        else:
            print('** Error: Can not create two controller before destory another! **')
            exit()

        # controll flag
        self.up = False
        self.down = False
        self.left = False
        self.right = False
        self.lshift = False
        self.lctrl = False
        self.record_button = False
        self.use_model = False

        # speed and angle init
        self.v = v
        self.a = a
        self.max_v = max_v
        self.rv_rate = rv_rate  # reverse speed rate
        self.speed = 0  # current speed
        self.angle = 0  # current angle
        self.record = False
        self.shutdown = False

        self.pressWait = time.time()

        pygame.display.set_caption("controller")
        size = 300
        self.screen = pygame.display.set_mode((size, size), pygame.RESIZABLE)
        co_filename = sys._getframe().f_code.co_filename
        path = co_filename[0: find_last(co_filename, '/') + 1] + 'images/th.jpeg'
        self.background = pygame.transform.scale(pygame.image.load(path), (size, size))
        self.screen.blit(self.background,(0,0))
        pygame.display.update()

    def render(self, speed = -99999999, angle = -99999999):
        if speed == -99999999:
            speed = self.speed
        if angle == -99999999:
            angle = self.angle

        font = pygame.font.SysFont('loma', 16, True)
        surface = font.render('speed = ' + str(speed), True, [0, 0, 0])
        surface2 = font.render('angle = ' + str(angle), True, [0, 0, 0])
        self.screen.blit(self.background,(0,0))
        self.screen.blit(surface, [20, 20])
        self.screen.blit(surface2, [20, 40])
        pygame.display.update()

    def listener(self):
        pygame.time.delay(10)
        # downing = -1 if self.down == True else 1
        downing = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT] and not self.left:
            self.left = True
            self.angle = self.a * downing
        elif not keys[pygame.K_LEFT] and self.left:
            self.left = False
            if self.right:
                self.angle = (- self.a) * downing
            else:
                self.angle = 0
 
        if keys[pygame.K_RIGHT] and not self.right:
            self.right = True
            self.angle = (- self.a) * downing
        elif not keys[pygame.K_RIGHT] and self.right:
            self.right = False
            if self.left:
                self.angle = self.a * downing
            else:
                self.angle = 0
 
        if keys[pygame.K_UP] and not self.up:
            self.up = True
            self.speed = self.v
        elif not keys[pygame.K_UP] and self.up:
            self.up = False
            self.speed = 0
 
        if keys[pygame.K_DOWN] and not self.down:
            self.down = True
            self.speed = - self.v * self.rv_rate
        elif not keys[pygame.K_DOWN] and self.down:
            self.down = False
            self.speed = 0

        if keys[pygame.K_LSHIFT] and not self.lshift:
            self.lshift = True
            if self.v + 0.05 < self.max_v:
                self.v += 0.05
            else:
                self.v = self.max_v
        elif not keys[pygame.K_LSHIFT] and self.lshift:
            self.lshift = False

        if keys[pygame.K_LCTRL] and not self.lctrl:
            self.lctrl = True
            if self.v - 0.05 >= 0:
                self.v -= 0.05
            else:
                self.v = 0
        elif not keys[pygame.K_LCTRL] and self.lctrl:
            self.lctrl = False

        if keys[pygame.K_r] and not self.record_button:
            self.record_button = True
            self.record = False if self.record else True
            if self.record:
                print('Start Recording')
            else:
                print('Record Stoped')
        elif not keys[pygame.K_r] and self.record_button:
            self.record_button = False
        
        if keys[pygame.K_l] and not self.use_model:
            if time.time() - self.pressWait >= 1:
                self.pressWait = time.time()
                self.use_model = True
                print('USING MODEL')
            
        elif keys[pygame.K_l] and self.use_model:
            if time.time() - self.pressWait >= 1:
                self.pressWait = time.time()

                self.use_model = False
                self.speed = 0
                self.angle = 0
                print("MODEL STOP")

        if keys[pygame.K_ESCAPE]:
            self.shutdown = True

    @classmethod
    def destory(cls):
        pygame.quit()
        cls.__Is_pygameinit = False

    @classmethod
    def Is_init(cls):
        return cls.__Is_pygameinit

    @classmethod
    def Inited(cls):
        cls.__Is_pygameinit = True

def find_last(str, key):
    pos = str.find(key)
    last = pos
    while pos != -1:
        pos = str[last + 1:].find(key)
        if pos != -1:
            last = last + pos + 1
    return last

if __name__ == '__main__':
    controller = KeyBoardController()

    while True:
        controller.listener()

    controller.destory()