import numpy as np
import cv2

from utils import CFEVideoConf, image_resize

cap = cv2.VideoCapture(0)

save_path           = 'saved-media/glasses_and_stash.mp4'
frames_per_seconds  = 60
config              = CFEVideoConf(cap, filepath=save_path, res='720p')
out                 = cv2.VideoWriter(save_path, config.video_type, frames_per_seconds, config.dims)
face_cascade        = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
eyes_cascade        = cv2.CascadeClassifier('cascades/third-party/frontalEyes35x16.xml')
glasses             = cv2.imread("images/fun/glasses.png", -1)
hat                 = cv2.imread('images/fun/hat.png',-1)

'''
OpenCV & Python Tutorial Video Series: https://kirr.co/ijcr59
Eyes Cascade (and others): https://kirr.co/694cu1
Nose Cascade / Mustache Post: https://kirr.co/69c1le
'''

eyes_prev = []
faces_prev = []
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces           = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)


    for (x, y, w, h) in faces if faces != () else faces_prev:
            roi_gray    = gray[y:y+w, x:x+h] # rec
            roi_color   = frame[y:y+w, x:x+h]

            roi_color_hat = frame[max(0, y -200):y+h, x:x+w]

            # eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)

            faces_prev = faces if faces != () else faces_prev

            try:
                hat2 = image_resize(hat.copy(), width = w)
                hw, hh, hc = hat2.shape
                for i in range(0, hw):
                    for j in range(0, hh):
                        if hat2[i, j][3] != 0:
                            roi_color_hat[i, j ] = hat2[i,j]

            except IndexError:
                pass

            # try:
            #
            #     for (ex, ey, ew, eh) in eyes if eyes != () else eyes_prev:
            #         #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
            #         glasses2 = image_resize(glasses.copy(), width=eh + 20)
            #         eyes_prev = eyes if eyes != () else eyes_prev
            #         gw, gh, gc = glasses2.shape
            #         for i in range(0, gw):
            #             for j in range(0, gh):
            #                 #print(glasses[i, j]) #RGBA
            #                 if glasses2[i, j][3] != 0: # alpha 0
            #                     roi_color[ey + i , ex + j ] = glasses2[i, j]
            # except IndexError:
            #     pass

    # Display the resulting frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    out.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
