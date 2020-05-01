class LeftEye():

	def __init__(self, shape, start, end):
		self._landmark_points = shape[start:end]

	
	def getLeftEye(self):
		return self._landmark_points



class RightEye():

	def __init__(self, shape, start, end):
		self._landmark_points = shape[start:end]

	
	def getRightEye(self):
		return self._landmark_points



class LeftEyebrow():

	def __init__(self, shape, start, end):
		self._landmark_points = shape[start:end]
	
	def getLeftEyebrow(self):
		return self._landmark_points



class RightEyebrow():

	def __init__(self, shape, start, end):
		self._landmark_points = shape[start:end]

	def getRightEyebrow(self):
		return self._landmark_points



class Mouth():

	def __init__(self, shape, start, end):
		self._landmark_points = shape[start:end]

	
	def getMouth(self):
		return self._landmark_points



class InnerMouth():

	def __init__(self, shape, start, end):
		self._landmark_points = shape[start:end]

	
	def getInnerMouth(self):
		return self._landmark_points



class Nose():

	def __init__(self, shape, start, end):
		self._landmark_points = shape[start:end]
	
	def getNose(self):
		return self._landmark_points



class Jawline():

	def __init__(self, shape, start, end):
		self._landmark_points = shape[start:end]
	
	def getJawline(self):
		return self._landmark_points
