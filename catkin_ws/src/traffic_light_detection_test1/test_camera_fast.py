
import cv2 as cv
from time import time

if __name__ == '__main__':


    #test camera with opencv
    cap = cv.VideoCapture(0)
    cv.namedWindow('test')
    cnt = 0

    #calculate fps
    fps = 0.0

    while True:
        prev_call = time()
        ret, frame = cap.read()
        cnt += 1
        if ret:
            cnt = 0
            cv.imshow('test', frame)
            key = cv.waitKey(1)
            if key == 27:
                cap.release()
                cv.destroyAllWindows()
                exit(0)
        if cnt > 100:
            cap.release()
            cv.destroyAllWindows()
            exit(0)
        loop_time = time() - prev_call
        fps = 1.0 / loop_time
        print("FPS: ", fps)