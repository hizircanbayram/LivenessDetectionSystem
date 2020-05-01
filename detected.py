import dlib
from imutils import face_utils

class Detected:

	def __init__(self, detector, frame):
		self._faces = []
		faces = detector(frame, 0)

		for face in faces:
			self._faces.append(Face(face, face_utils.rect_to_bb(face)))


	def getFaces(self):
		return self._faces


	def getLength(self):
		return len(self._faces)





class Face:

	def __init__(self, detected_face, bounding_box):
		self._detected_face = detected_face
		self._bounding_box = bounding_box		


	def getDetectedFace(self):
		return self._detected_face


	def getFaceBB(self):
		(x, y, w, h) = face_utils.rect_to_bb(self._detected_face)
		return (x, y), (x + w, y + h)

