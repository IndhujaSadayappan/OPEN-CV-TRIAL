import cv2
import time
from plyer import notification


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


cap = cv2.VideoCapture(0)


BLINK_THRESHOLD = 2   
CONSECUTIVE_FRAMES = 20 
frame_count = 0         
blink_start_time = time.time()  

# Posture detection parameters
posture_start_time = time.time()  
face_position = None              
NO_MOVEMENT_DELAY = 30            

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))


    eyes_detected = False

   
    eye_frame = frame.copy()
    posture_frame = frame.copy()

    for (x, y, w, h) in faces:
       
        cv2.rectangle(posture_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

     
        current_position = (x, y, w, h)
        if face_position is None:
            face_position = current_position  

        if abs(face_position[0] - x) > 10 or abs(face_position[1] - y) > 10:
            posture_start_time = time.time()  
            face_position = current_position  

      
        face_region = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))

        if len(eyes) >= 2:  
            eyes_detected = True
            frame_count = 0       
            blink_start_time = time.time() 

        else:
            frame_count += 1 

       
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(eye_frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

   
    if time.time() - blink_start_time > 30:
        notification.notify(
            title="Blink Reminder",
            message="Please blink to prevent eye strain!",
            timeout=5
        )
        blink_start_time = time.time()  

   
    if time.time() - posture_start_time > NO_MOVEMENT_DELAY:
        notification.notify(
            title="Posture Alert",
            message="Adjust your posture to prevent strain!",
            timeout=5
        )
        posture_start_time = time.time()  


    cv2.imshow("Eye Detection Frame", eye_frame)
    cv2.imshow("Posture Detection Frame", posture_frame)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
