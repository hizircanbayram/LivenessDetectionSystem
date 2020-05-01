import cv2
from scipy.spatial import distance as dist


class EyeBlinkSpec():

    def __init__(self, landmark_detection):
        self._EAR_THRESH = 0.30
        self._landmark_detection = landmark_detection
        self._prev_ear = 0
        self._ear = 0
        self._eye_blink_num = 0   
        self._prev_frame_closed = False


    def setPrevFrameClosed(self, state):
        self._prev_frame_closed = state


    def increaseEyeBlinkNum(self):
        self._eye_blink_num += 1


    def getPrevFrameClosed(self):
        return self._prev_frame_closed     


    def setLandmarkDetection(self, landmark_detection):
        self._landmark_detection = landmark_detection

	
    def _eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])

        ear = (A + B) / (2.0 * C)
        return ear


    def getEyeBlinkNum(self):
        return self._eye_blink_num


    def setEyeBlinkNum(self, num):
        self._eye_blink_num = num

            
    def isClosed(self):
        leftEye = self._landmark_detection.getLeftEye()
        rightEye = self._landmark_detection.getRightEye()
        leftEAR = self._eye_aspect_ratio(leftEye)
        rightEAR = self._eye_aspect_ratio(rightEye)
        self._prev_ear = self._ear
        self._ear = (leftEAR + rightEAR) / 2.0
        # ayri bir 0.05 esik degeri eklenmediginde ve goz kirpik birakildiginda en ufak dalgalanmalarda gozun kirpildigini zannediyor.
        # bu esik degeri sayesinde gozun acik ve kapali olma state'leri birbirine cok bagli olmuyor. aralarina mesafe yerlestiriliyor. 
        if self._ear < self._EAR_THRESH - 0.05:   
            return True
        elif self._ear > self._EAR_THRESH + 0.05:
            return False     


    def isBlinked(self, face_stabler):
        current_frame_closed = self.isClosed()
        if self._prev_frame_closed and (not current_frame_closed) and face_stabler.isStable():
            self.increaseEyeBlinkNum()
        self._prev_frame_closed = current_frame_closed


