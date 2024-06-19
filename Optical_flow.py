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
                old_frame=self.draw_cross(old_frame,self.nose_point)
                old_frame=cv2.resize(old_frame,(800,700))
                #self.mask = np.zeros_like(old_frame)
                cv2.imshow('Nose',old_frame)

                self.key=cv2.waitKey() 
                if self.key== 32:
                    self.Msg="<span style='color: #RRGGBB;'>Permanently Closed,Load the video again ...</span>"
                    cv2.destroyAllWindows()
                    video.release()
                    
    
                elif self.key==ord('q'):
                    self.Msg="<span style='color: #00008B;'>Partially Closed ...</span>" 
                    cv2.destroyAllWindows()
                    
                        
    def track_and_display(self, video):
        track = []
        ret, frame = video.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.p0 = cv2.goodFeaturesToTrack(gray_frame, maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7)

            if self.p0 is not None:
                self.nose_point = tuple(map(int, self.p0[0][0]))

            frame = self.draw_cross(frame, self.nose_point)
            track.append(self.nose_point)
            self.old_gray = gray_frame.copy()

        while True:
            ret, frame = video.read()
            if not ret:
                self.Msg="<span style='color: #00008B;'>End of the Stream ...</span>"
                cv2.destroyAllWindows()
                break


            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, gray_frame, self.p0, None)

            if p1 is not None and st.any():
                self.nose_point = tuple(map(int, p1[0][0]))
                track.append(self.nose_point)

            frame = self.draw_cross(frame, self.nose_point)
            for i in track:
                cv2.circle(frame, i, 1, (0, 255, 255), 3)

            frame=cv2.resize(frame,(800,700))
            cv2.imshow('Tracking', frame)
            self.old_gray = gray_frame.copy()
            self.p0 = p1
      

            self.key_2=cv2.waitKey(30)
            if self.key_2== 32:
                self.Msg="<span style='color: #RRGGBB;'>Permanently Closed,You could load again the video...</span>"
                cv2.destroyAllWindows()
                video.release()
                break
  
            elif self.key_2==ord('q'):
                self.Msg="<span style='color: #00008B;'>Partially Closed ...</span>"  
                cv2.destroyAllWindows()
                break

        
    def draw_cross(self, frame, point):
        cv2.line(frame, (point[0] - 10, point[1]), 
                (point[0] + 10, point[1]), (0, 0, 255), 4)
        cv2.line(frame, (point[0], point[1] - 10), 
                (point[0], point[1] + 10), (0, 0, 255), 4)
        return frame
