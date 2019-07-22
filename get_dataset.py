import configparser
import cv2 as cv
import numpy as np


config = configparser.ConfigParser()
print('...loading from config.ini')
config.read('config.ini')
test_video_name = config.get('data_set','test_video_name')

cap = cv.VideoCapture(test_video_name)
delay = 10
pause_flag = False
rat, frame = cap.read()
rect = [0,0,1,1]


def onMouse(event,x,y,flags,param):
    global rect
    if event == cv.EVENT_LBUTTONDOWN:
        print('happened')
        rect[0], rect[1] = x, y
    elif event == cv.EVENT_LBUTTONUP:
        rect[2], rect[3] = x, y

cv.namedWindow('frame')
cv.setMouseCallback('frame',onMouse)

while cap.isOpened():
    if not pause_flag:
        rat, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    print(rect)
    cv.rectangle(gray,(rect[0],rect[1]), (rect[2],rect[3]), (255, 0, 0), 2, -1)
    cv.imshow('frame',gray)
    opt = cv.waitKey(delay) & 0xff
    if opt == ord('q'):
        break
    elif opt == ord('s'):
        pause_flag = not pause_flag
for i in range(4):
    config.set('data_set','rectangle_pos%i' % i, '%i' % rect[i])

input()
cap.release()
cv.destroyAllWindows()

config.write(open('config.ini','w'))





