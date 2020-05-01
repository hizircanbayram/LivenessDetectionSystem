# LivenessDetectionSystem

This software checks if the person in front of a camera is alive or not using traditional techniques. 

## Main Tasks The Person in Front of a Camera Can Do:
  * Blink his/her eyes for the desired amount of time in the given duration.
  * To have a desired emotion in his/her face for the given duration.
  * Move his/her head where it is desired for the given duration.

## How to Notify
If all of the desired actions are done successfully, then the user is notified with a voicemail and program is terminated.
If one of them can't be done, then fresh three actions are assigned to the person and a buzz voice is played.
  
## Introducing the Current Data  
* User's head's 3-dimensional angles are shown at the upper left section the screen.
* Current task is shown at the top of his/her head.
* His current success is shown at the above of his/her head and right below his/her current task.
* All his/her assigned tasks are shown at the lower left section of the screen.
Note: If the current task is eye blinking, then the user must stable his/her head according to directions popped up at the top of the screen. If not, then the eye blinking does not count. His/her current head angle and the range it is supposed to be in are shown in the message.

![unknown](https://user-images.githubusercontent.com/23126077/80846774-16541600-8c16-11ea-8f39-e6ce596dfb98.png)

## Prerequisites
* cv2
* dlib
* imutils
* numpy
* scipy
 
## Built With
* [Python](https://www.python.org/)
