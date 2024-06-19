import cv2 as cv

class BackgroundSub:
    def __init__(self):
        self.Endmsg="<span style='color: #00008B;'>End of stream...</span>"
    

    def Background(self,video):
        #cap=cv.VideoCapture(video_path)

        #Initializes a KNN background subtractor.
        fgbg = cv.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)
        #frame_number = 0

        while True:
            ret,frame= video.read()
            if not ret:
                print(self.Endmsg)
                break

            
            blur_frame=cv.GaussianBlur(frame,(5,5,),0) #Gaussian blur
            fgmask= fgbg.apply(blur_frame) #foreground Binary mask
            #Extracts the moving objects by applying the fgmask to the original frame.
            foreground=cv.bitwise_and(frame,frame,mask=fgmask) 

            cv.imshow('Rgb Frame',frame)
            cv.imshow('Foreground Mask',fgmask)
            cv.imshow('Result',foreground)

            key=cv.waitKey(30)

            if key == 32:
                break

        video.release()
        cv.destroyAllWindows()
                








        

       
        
        

        