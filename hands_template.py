#imports
import math

import cv2
import mediapipe as mp
import time

class Hands_detection():
    def __init__(self,mode=False,maxhands=2,detection_confidence=0.5,tracking_confidence=0.5):
        self.mode = mode
        self.maxhands= maxhands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode,self.maxhands,self.detection_confidence,self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tip_id = [4,8,12,16,20]

    def hands_detect(self,img,draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(img_rgb)
        # print(result.multi_hand_landmarks)

        if self.result.multi_hand_landmarks:
            for land in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, land, self.mphands.HAND_CONNECTIONS)

            return img

    def find_locations(self,img,handNo=0,draw=True):
        self.landmarks = []
        x_list=[]
        y_list=[]
        if self.result.multi_hand_landmarks:
            my_hand = self.result.multi_hand_landmarks[handNo]
            for id, loc in enumerate(my_hand.landmark):
                h, w = img.shape[:2]
                px, py = int(loc.x * w), int(loc.y * h)
                x_list.append(px)
                y_list.append(py)
                self.landmarks.append([id,px,py])
                if draw:
                    cv2.circle(img,(px,py),5,(255,0,255),cv2.FILLED)

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            box = xmin, ymin, xmax, ymax
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color=(0,255,0),thickness=1)
            return  self.landmarks,box

    def finger_Up(self):
        fingers = []
        # for thumb checking closing with respect ot the x axis, currently for left hand
        if self.landmarks[self.tip_id[0]][1] < self.landmarks[self.tip_id[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):

            if self.landmarks[self.tip_id[id]][2] < self.landmarks[self.tip_id[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(self,p1,p2,img,draw=True,r=15,t=3):
        x1,y1 = self.landmarks[p1][1:]
        x2,y2 = self.landmarks[p2][1:]
        cx,cy = (x1+x2)//2,(y1+y2)//2

        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,0,255),t)
            cv2.circle(img,(x1,y1),r,(255,0,255),cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        le = math.hypot(x2-x1,y2-y1)

        return le,[x1,y1,x2,y2,cx,cy]


def main():
    frame = cv2.VideoCapture(0)
    previous_time = 0
    current_time = 0
    detector = Hands_detection()
    while True:
        _, img = frame.read()

        imk = detector.hands_detect(img)
        locations = detector.find_locations(imk)
        finger = detector.finger_Up()
        if len(locations) != 0:
            #print(locations[4])
            print(finger)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(imk, str(int(fps)), (20, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)

        cv2.imshow("Image",img)
        cv2.waitKey(1)


if __name__=='__main__':
    main()

































