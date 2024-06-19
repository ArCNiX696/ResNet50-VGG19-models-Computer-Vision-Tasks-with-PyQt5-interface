import cv2
import numpy as np

class Optical:
    def __init__(self):
        self.Msg = ''
        self.p0 = None
        self.mask = None
        self.color = np.random.randint(0, 255, (100, 3))

    def Preprocess(self, video):
        ret, old_frame = video.read()
        if ret:
            gray_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            self.p0 = cv2.goodFeaturesToTrack(gray_frame, maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7)
            if self.p0 is not None:
                self.nose_point = tuple(map(int, self.p0[0][0]))
                self.mask = np.zeros_like(old_frame)
            self.old_gray = gray_frame

    def track_and_display(self, video):
        track = []
        if self.nose_point:
            track.append(self.nose_point)
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, gray_frame, self.p0, None)
            
            if p1 is not None and st.any():
                self.nose_point = tuple(map(int, p1[0][0]))
                track.append(self.nose_point)
                for i in track:
                    cv2.circle(frame, i, 1, (0, 255, 255), 3)
                self.old_gray = gray_frame.copy()
                self.p0 = p1

            cv2.imshow('Tracking', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()