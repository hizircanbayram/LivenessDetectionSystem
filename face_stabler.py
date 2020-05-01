import cv2


class FaceStabler():

    def __init__(self, ranges):
        self.x_range = ranges[0]
        self.y_range = ranges[1]
        self.z_range = ranges[2]
        self.x_stable = False
        self.y_stable = False
        self.z_stable = False
        self.stable = False

    def warningLog(self, frame, angles):
        x = angles[0]
        y = angles[1]
        z = angles[2]

        if x < self.x_range[0]:
            cv2.putText(frame, "Kafanizi asagi egin        " + "[" + str(self.x_range[0]) + "," + str(self.x_range[1]) + "]" , (160, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)
            cv2.putText(frame,  "{:7.2f}".format(x), (540, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)
            self.x_stable = False       
        elif x > self.x_range[1]:
            cv2.putText(frame, "Kafanizi yukari kaldirin   " + "[" + str(self.x_range[0]) + "," + str(self.x_range[1]) + "]" , (160, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)   
            cv2.putText(frame,  "{:7.2f}".format(x), (540, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)   
            self.x_stable = False
        else:  
            self.x_stable = True

        if y < self.y_range[0]:
            cv2.putText(frame, "Yuzunuzu saga cevirin  " + "[" + str(self.y_range[0]) + "," + str(self.y_range[1]) + "]" , (160, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)    
            cv2.putText(frame,  "{:7.2f}".format(y), (540, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)  
            self.y_stable = False      
        elif y > self.y_range[1]:
            cv2.putText(frame, "Yuzunuzu sola cevirin  " + "[" + str(self.y_range[0]) + "," + str(self.y_range[1]) + "]" , (160, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)      
            cv2.putText(frame,  "{:7.2f}".format(y), (540, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)    
            self.y_stable = False
        else:
            self.y_stable = True

        if z < self.z_range[0]:
            cv2.putText(frame, "Kafanizi sola cevirin  " + "[" + str(self.z_range[0]) + "," + str(self.z_range[1]) + "]" , (160, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)  
            cv2.putText(frame,  "{:7.2f}".format(z), (540, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)             
            self.z_stable = False
        elif z > self.z_range[1]:
            cv2.putText(frame, "Kafanizi saga cevirin  " + "[" + str(self.z_range[0]) + "," + str(self.z_range[1]) + "]" , (160, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)      
            cv2.putText(frame,  "{:7.2f}".format(z), (540, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)
            self.z_stable = False    
        else:     
            self.z_stable = True


    def isStable(self):
        if self.x_stable and self.y_stable and self.z_stable:
            return True
        else:
            return False

