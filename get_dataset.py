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


def onChange(emp):
    pass


cv.namedWindow('video')
cv.setMouseCallback('video',onMouse)
frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
font = cv.FONT_HERSHEY_SIMPLEX

cv.createTrackbar('frame', 'video', 0, frame_count, onChange)
print('...display video')
pos = 0
last_pos = 0
index = 0
label = np.zeros(frame_count,dtype=np.int8)
while cap.isOpened():
    pos = cv.getTrackbarPos('frame','video')
    if not pause_flag:
        if pos != last_pos+1:
            cap.set(cv.CAP_PROP_POS_FRAMES,pos)
        rat, frame = cap.read()
        index = pos
        last_pos = pos
        pos = pos + 1
        cv.setTrackbarPos('frame','video',pos)
    else:
        if pos != last_pos:
            cap.set(cv.CAP_PROP_POS_FRAMES,pos)
        if pos != last_pos :
            rat, frame = cap.read()
            index = pos
        last_pos = pos
        pass

    opt = cv.waitKey(delay) & 0xff
    if pos == frame_count:
        pause_flag = 1
        continue
    # img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = frame.copy()
    cv.putText(img,'index:%i' % index,(0,25),font,1,(255,255,255))
    cv.putText(img,'label:%i' % label[index],(0,80),font,2,(255,255,255) if label[index] else (0,0,0),2)
    # print(rect)
    cv.rectangle(img,(rect[0],rect[1]), (rect[2],rect[3]), (255, 0, 0), 2, -1)
    cv.imshow('video',img)
    if opt == ord('q') :
        break
    elif opt == ord('s'):
        pause_flag = not pause_flag
    elif opt == ord('a'):
        pos = pos - 1
        cv.setTrackbarPos('frame','video',pos)
    elif opt == ord('d'):
        pos = pos + 1
        cv.setTrackbarPos('frame','video',pos)
    elif opt == ord('w'):
        label[index] = 1 - label[index]
        img = frame
        cv.putText(img,'index:%i' % index,(0,25),font,1,(255,255,255))
        cv.putText(img,'label:%i' % label[index],(0,80),font,2,(255,255,255) if label[index] else (0,0,0),2)


for i in range(4):
    config.set('data_set','rectangle_pos%i' % i, '%i' % rect[i])

cap.release()
cv.destroyAllWindows()

config.write(open('config.ini','w'))







