###
#   show the output with vedio
###

import os
import time
import pygame
import math
import json
from PIL import Image

class sViewr():
    def __init__(self, width, height):
        pygame.init()
        self.display=pygame.display.set_mode((width,height), 0)

    def update(self):
        pass
    
    def run(self,path, angle, throttle, real_angel, real_throttle):
        if angle!=None:
            img = pygame.image.load(path)
            
            #red line for virtual angle
            #print(angle)
            angle = angle * 1.2
            real_angel = real_angel * 1.2
            
            x=128+30*math.sin(angle/1*math.pi/4)
            y=100+20-30*math.cos(angle/1*math.pi/4)
            pygame.draw.line(img, (255,0,0), (128,144), (x,y), 3)
            myfont = pygame.font.Font(None, 20)
            pThrottle = myfont.render(str(throttle), True, (255,0,0))

            string=""
            for i in path.split('/')[:-1]:
                string+=i+"/"
                # print(string)
            snum=path.split('/')[-1].split('_')[0]

            rThrottle = myfont.render(str(real_throttle), True, (0,255,0))
            
            #green line for real angle 
            rx=128+20*math.sin(real_angel/1*math.pi/4)
            ry=100+20-20*math.cos(real_angel/1*math.pi/4)
            pygame.draw.line(img, (0,255,0), (128,144), (rx,ry), 3)
            
            #pygame.image.save(img,"outputdata/"+snum+".jpeg")
            self.display.blit(img, (0,0))
            self.display.blit(pThrottle, (128,0))
            self.display.blit(rThrottle, (0,0))
            time.sleep(0.1)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
    
    def shutdown(self):
        pygame.quit()
        pass