from detected import Face
import dlib
from imutils import face_utils
from facial_organs import LeftEye, RightEye, LeftEyebrow, RightEyebrow, Nose, Jawline, Mouth, InnerMouth
import cv2



class LandmarkDetection():

    predictor = dlib.shape_predictor("shape_predictor.dat")
    
    def __init__(self, frame, face):
        shape = LandmarkDetection.predictor(frame, face)
        shape = face_utils.shape_to_np(shape)
        self._landmark_points = shape
        self.set_inits()


    def __init__(self, shape):
        self._landmark_points = shape
        self.set_inits()



    def set_inits(self):
        leftEyeStart, leftEyeEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        rightEyeStart, rightEyeEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        leftEyebrowStart, leftEyebrowEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
        rightEyebrowStart, rightEyebrowEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
        mouthStart, mouthEnd = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        innerMouthStart, innerMouthEnd = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]    
        noseStart, noseEnd = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
        jawStart, jawEnd = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

        self._left_eye = LeftEye(self._landmark_points, leftEyeStart ,leftEyeEnd)
        self._right_eye = RightEye(self._landmark_points, rightEyeStart, rightEyeEnd)

        self._left_eyebrow = LeftEyebrow(self._landmark_points, leftEyebrowStart, leftEyebrowEnd)
        self._right_eyebrow = RightEyebrow(self._landmark_points, rightEyebrowStart, rightEyebrowEnd)

        self._mouth = Mouth(self._landmark_points, mouthStart, mouthEnd)
        self._inner_mouth = InnerMouth(self._landmark_points, innerMouthStart, innerMouthEnd)

        self._nose = Nose(self._landmark_points, noseStart, noseEnd)
        self._jawline = Jawline(self._landmark_points, jawStart, jawEnd)

    def getSpecificTuple(self, i):
        '''
        Returns the specific facial landmark tuple based on the given integer number.
        Param | i : The index number of facial landmark
        Return | : (x,y) coordinates of the specific facial landmark based on the current frame
        '''
        return (self._landmark_points[i][0], self._landmark_points[i][1])


    def getLeftEye(self):
        return self._left_eye.getLeftEye()


    def getRightEye(self):
        return self._right_eye.getRightEye()


    def getLeftEyebrow(self):
        return self._left_eyebrow.getLeftEyebrow()


    def getRightEyebrow(self):
        return self._right_eyebrow.getRightEyebrow()


    def getMouth(self):
        return self._mouth.getMouth()


    def getInnerMouth(self):
        return self._inner_mouth.getInnerMouth()
    
    
    def getNose(self):
        return self._nose.getNose()

    
    def getJawline(self):
        return self._jawline.getJawline()
