import cv2
import autopy
import mediapipe as mp
import numpy as np

screen_width, screen_height = autopy.screen.size()

mp_hands=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils
hands=mp_hands.Hands(min_detection_confidence=0.75, max_num_hands=1)

cap = cv2.VideoCapture(0)
# url = 'http://192.0.0.4:8080/video'  #url of the ip webcam app
# cap = cv2.VideoCapture(url)

camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def process_image(img):
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting BGR to RGB since mp_hands takes only RGB images
    outputs = hands.process(RGB_img)
    return outputs


def find_hands(img, outputs):
    if outputs.multi_hand_landmarks:
        for single_hand_landmarks in outputs.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, single_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return img

def find_handlandmark_position(img, outputs, hand_no=0):
    landmark_list = []
    if outputs.multi_hand_landmarks:
        my_hand = outputs.multi_hand_landmarks[hand_no]
        for id, land_mark in enumerate(my_hand.landmark):
            h, w, c = img.shape
            cx, cy = int(land_mark.x * w), int(land_mark.y * h)   #converting normalized values to pixel values
            landmark_list.append([id, cx, cy])
            
   
    if landmark_list:
        landmark_array = np.array(landmark_list)
        min_x, min_y = np.min(landmark_array[:, 1]), np.min(landmark_array[:, 2])
        max_x, max_y = np.max(landmark_array[:, 1]), np.max(landmark_array[:, 2])
        bounding_box = [min_x, min_y, max_x, max_y]
    else:
        bounding_box = [None, None, None, None]
        
    return landmark_list, bounding_box


while True:
    success, img = cap.read()
    outputs = process_image(img)
    img = find_hands(img, outputs)
    LandMarkList,boundingbox = find_handlandmark_position(img, outputs)
    
    if len(LandMarkList)!=0:
        id_x, id_y = LandMarkList[8][1], LandMarkList[8][2]
        thumb_x, thumb_y = LandMarkList[4][1], LandMarkList[4][2]
        middle_x, middle_y = LandMarkList[12][1], LandMarkList[12][2]
        cv2.circle(img, (id_x, id_y), 15, (255, 255, 255), cv2.FILLED)
        cv2.circle(img, (thumb_x, thumb_y), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (middle_x, middle_y), 15, (255, 0, 255), cv2.FILLED)
        
        dist=np.hypot(middle_x-thumb_x,middle_y-thumb_y)
        print(dist)
        if dist<65:
            cv2.circle(img, (id_x, id_y), 15, (0, 255, 0), cv2.FILLED)
            new_x=np.interp(middle_x,(0,camera_width),(0,screen_width))
            new_y=np.interp(middle_y,(0,camera_height),(0,screen_height))
            new_x = max(0, min(new_x, screen_width - 1))
            new_y = max(0, min(new_y, screen_height - 1))
            autopy.mouse.move(new_x,new_y)
        else:
            cv2.circle(img, (middle_x, middle_y), 15, (0, 0, 255), cv2.FILLED)
            autopy.mouse.click()
        
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break