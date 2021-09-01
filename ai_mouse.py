import cv2
import mediapipe as mp
import numpy as np
import time
import hands_template as ht
import autopy

h,w = 480,640
h_s,w_s = autopy.screen.size()
frame = cv2.VideoCapture(0)
frame.set(3,w)
frame.set(4,h)
ctime = 0
ptime = 0
frameR = 100
smooth = 15
prev_loc_x,prev_loc_y = 0,0
curr_loc_x,curr_loc_y = 0,0




detector = ht.Hands_detection(maxhands=1)

while True:

    _ , img = frame.read()
    # 1. find hand landmarks
    imk = detector.hands_detect(img)
    locations,box= detector.find_locations(imk)


    # 2. get tips of index and middle fingers
    if len(locations) != 0:
        x1, y1 = locations[8][1:]
        x2, y2 = locations[12][1:]

        #print(x1,y1,x2,y2)
    # 3. check fingers up?
    fingers = detector.finger_Up()
    print(fingers)
    # 4. In moving mode(index finger)?
    cv2.rectangle(img, (frameR, frameR), (w - frameR, h - frameR), (255, 0, 255), 2)
    if fingers[1]==1 and fingers[2]==0:
        #5. Convert coordinates

        # scaling the mouse pointer movement on screen w.r.t rectangle
        x3 = np.interp(x1,(frameR,w-frameR),(0,w_s))
        y3 = np.interp(y1,(frameR,h-frameR),(0,h_s))

        # 6.Smoothen values
        curr_loc_x = prev_loc_x + (x3 - prev_loc_x) / smooth
        curr_loc_y = prev_loc_y + (y3 - prev_loc_y) / smooth

        #7. move mouse

        autopy.mouse.move(w_s-curr_loc_x,curr_loc_y)
        cv2.circle(img,(x1,y1),25,(255,0,255),cv2.FILLED)
        prev_loc_x = curr_loc_x
        prev_loc_y = curr_loc_y


    #8. check if clicking mode (index and middle up)
    if fingers[1] == 1 and fingers[2] == 1:
        # 9. find distance b/w fingers
        length , line_info = detector.find_distance(8,12,img)
        print(length)
        # 10. if dist is short click mouse.
        if length<35:
            cv2.circle(img,(line_info[4],line_info[5]),15,(0,255,0),cv2.FILLED)
            autopy.mouse.click()





    #11. Frame rate(fps)
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img, str(int(fps)), (20, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)

    #12. Display
    cv2.imshow("Output",img)
    cv2.waitKey(1)

