import cv2
import time
import dlib
import random
import imutils
import argparse
import numpy as np
from random import randrange
from threading import Thread
from imutils import face_utils
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from scipy.spatial import distance as dist

from face_stabler import FaceStabler
from get_head_pose import get_head_pose
from eye_blink_spec import EyeBlinkSpec
from landmark_detection import LandmarkDetection
from emotion_detector.main_predictor import EmotionClassifier



def write_actions(actions):
    if actions[0].getDone() == True:
        cv2.putText(frame, actions[0].getAction() + str(actions[0].getOccurrenceNum()), (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness=2)
    else:
        cv2.putText(frame, actions[0].getAction() + str(actions[0].getOccurrenceNum()), (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)
    if actions[1].getDone() == True:
        cv2.putText(frame, actions[1].getAction() + str(actions[1].getOccurrenceNum()), (20, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness=2)
    else:
        cv2.putText(frame, actions[1].getAction() + str(actions[1].getOccurrenceNum()), (20, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)
    if actions[2].getDone() == True:
        cv2.putText(frame, actions[2].getAction() + str(actions[2].getOccurrenceNum()), (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness=2)
    else:
        cv2.putText(frame, actions[2].getAction() + str(actions[2].getOccurrenceNum()), (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)


def sound_alarm(path):
    import pygame
    import time
    pygame.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    time.sleep(4)



class LivenessDetection():

    class Action():
        #Kullaniciya atanan her gorev bir Action'dir. 5 kere blinking, 3 sn yukari bakma gibi.
        #self._action : blinking, happy, neutral, up, down etc.
        #self._label : blinking, direction, emotion.
        def __init__(self, action):
            (self._action, self._occurrence_num) = action.split('/')
            self._done = False
            self._occurrence_num = int(self._occurrence_num)
            if self._action in LivenessDetection.blinking_literal:
                self._label = 'blinking'
            elif self._action in LivenessDetection.direction_literal:
                self._label = 'direction'
            elif self._action in LivenessDetection.emotion_literal:
                self._label = 'emotion'
            else:
                print('UNDEFINED LABEL FOR ACTION')
                self._label = 'UNDEFINED'


        def getLabel(self):
            return self._label

        def getAction(self):
            return self._action

        def getDone(self):
            return self._done

        def getOccurrenceNum(self):
            return self._occurrence_num

        def setLabel(self, label):
            self._label = label

        def setAction(self, action):
            self._action = action

        def setDone(self, done):
            self._done = done

        def setOccurrenceNum(self, num):
            self._occurrence_num = num
            


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor.dat')
    # buradan hareketler kisitlanabilir ya da artirilabilir. mesela emotio_literal'e 'angry' eklenebilirken, direction_literal'den 'up' cikarilabilir.
    blinking_literal = ['blinking']
    direction_literal = ['up', 'down', 'right', 'left']
    emotion_literal = ['happy', 'neutral']
    #emotion_literal = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']

    def __init__(self):
        # hyperparameters
        self._action_num = 3 # kullanicinin kac hareket yapmasini istedigini belirledigimiz sayi.
        self._validness_threshold = 10 # hareketi dogru yapmaya basladiktan sonra eger self._validnesss_threshold kadar frame'de dogru hareketi yapmazsa hareket basarisiz sayilir
        self._blinking_sec = 20 # uretilen goz kirpma sayisini kadar kirpma islemini self._blinking_sec saniyede yapmazsa hareket basarisiz sayilir
        # parameters
        self._actions_to_do = [] # kullanicinin yapacagi action'lar burada saklanir. ornek: (blinking 5), (up 3), (happy 4)
        self._combinations = [] # verilen parametrelere bagli tum olasi kombinasyonlar bu listede toplanir, random olarak bunlardan hareketler secilir ve self._action_to_do'ya atabir.
        self._addAction(self.blinking_literal, range(2,7)) # goz kirpma hareketini ekler. range ile ayarlanan, kac adet goz kirpma kombinasyonunun olabilecegi
        self._addAction(self.direction_literal, (3,5)) # herhangi yone bakma hareketini ekler. range ile ayarlanan, kac saniye ilgili hareketin yapilacagi kombinasyonun olabilecegi
        self._addAction(self.emotion_literal, [2,3,4]) # herhangi duygu durumunu ekler. range ile ayarlanan, kac saniye ilgili duygu durumunda durulacaginin kombinasyonu
        self._generate_actions_todo()
        self._face_stabler = FaceStabler(([0,10],[-15,15],[-10,10])) # goz kirpma esnasinda kafa pozu burada verilen araligin disina ciktiginda goz kirpma artirilmiyor ve kullaniciya kafasini tekrar
        # istenilen acilara sokmak icin yonlendirmeler yapiliyor.
        self._eye_blink = EyeBlinkSpec(None) # goz kirpma sayisini sayan sinif
        self._access_granted = False # tum action'lar basariyla tamamlandiginda True oluyor ve program sonlandiriliyor.
        self._emotion_detector = EmotionClassifier() # duygu tahminin yapan sinif
        # counters
        self._success_state_counter = 0 # ilgili action tamamlandikca bir artirilir. self._action_num'a esit oldugunda program durdurulur(action'lar basariyla tamamlanmistir)
        self._start = None # action dogru yapilmaya baslandiginda artik None degildir
        self._validness_counter = 0 # action dogru yapilmaya baslandiktan sonra yanlis action'in yapilan her frame'de 1 artirilir. eger bu yanlis action frame'lerinin sayisi self._validness_threshold
        # degiskenini asarsa ilgili action basarisiz sayilir.
        self._start_timer = 0 # hareket dogru yapilmaya baslandiktan sonra bu degisken gecen saniyeyi tutar. 
        self._state = self._actions_to_do[self._success_state_counter] # ilk hareketi mevcut state olarak atar.
        # the upper left corner of the head        
        self._x = None
        self._y = None


    def getAccessGranted(self):
        return self._access_granted


    def _addAction(self, cases, repeated_num):
        all_combinations = []
        for i1 in cases: # cases degiskeni yukaridkai ***_literal dizileri ile doldurulur. mesela up, down, right, left.
            for i2 in repeated_num: # burada ise ilgili hareketin ne kadar sure / kac kere yapilacagi bilgisi tutulur. mesela 2,3,4,5.
                one_comb = i1 + '/' + str(i2)
                all_combinations.append(one_comb) 
        self._combinations.extend(all_combinations) # hepsi self._combinations'a eklenir ki self._actions_to_do bunlarin arasindan secim yapilarak doldurulabilsin


    def _generate_actions_todo(self):
        # tum kombinasyonlar icinden self._actions_num kadar action uretir. bunu yaparken bir turda(access granted olana kadar yapilacak task sayisi) ayni task'lerin art arda gelmemesi icin
        # recursive bir cagridan faydalanir
        self._actions_to_do = []
        random.shuffle(self._combinations)
        found = False
        for i in range(self._action_num):
            possible_action = self.Action(self._combinations[randrange(len(self._combinations))])
            for action in self._actions_to_do:
                if possible_action.getAction() == action.getAction():
                    found = True
            if found == True:
                self._generate_actions_todo()
            else:
                self._actions_to_do.append(possible_action)


    def _zeroCounter(self):
        # herhangi sebepten dolayi 'access denied' olursa gerekli degiskenleri sifirlar ve sistemin yeni gorevler icin tekrar calisabilmesini saglar
        self._validness_counter = 0
        self._success_state_counter = 0
        self._actions = self._generate_actions_todo()
        self._start = None
        self._start_timer = 0
        self._state = self._actions_to_do[self._success_state_counter]
        self._eye_blink.setEyeBlinkNum(0)    
        t = Thread(target=sound_alarm, args=("wrong.mp3",))   
        t.deamon = True
        t.start()   



    def _changeState(self):
        # ilgili task'lerden biri basarilir olunca yeni state'e gecmek icin gerekli olan yurutmeleri yapar.
        self._actions_to_do[self._success_state_counter].setDone(True) # mevcut task'in basarili sekilde yapildigi isaretleniyor
        self._success_state_counter += 1 # bir sonraki state'in counter'i olusturuluyor.
        # asagidaki dort satirda gerekli degiskenler sifirlaniyor.
        self._start_timer = 0
        self._start = None
        self._validness_counter = 0
        self._eye_blink.setEyeBlinkNum(0)   
        if self._success_state_counter < self._action_num: # eger tum task'ler bitmemisse
            self._state = self._actions_to_do[self._success_state_counter] # yeni task'i state'e alir
            t = Thread(target=sound_alarm, args=("successful.wav",))          
            t.deamon = True
            t.start()



    def _startTimer(self):
        # bir action dogru yapildiginda zamanlayiciyi baslatir. ardindan ilgili action'a verilmis zaman boyunca esik degerinin ustunde yanlis frame yapilmamissa ve buradaki zamanlayici
        # ilgili action icin verilen zamani doldurmussa ilgili action basariyla tamamlanmis olur.
        if self._start_timer < 2:
            self._start_timer += 1
        if self._start_timer == 1:
            self._start = time.time()  



    def is_looking_right_place(self, angles, frame): 
        # ilgili action'un label'ina gore ve hareketi dogru yapip yapmamasina gore True ya da False dondurur. ayrica harekete dair bilgilendirme yazisi yazar. 
        (angle_X, angle_Y, angle_Z) = angles
        if self._state.getAction() == 'up':
            if angle_X < -15:
                cv2.putText(frame, "up", (self._x, self._y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)
                return True
            else:
                cv2.putText(frame, "nothing", (self._x, self._y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)
                return False
        elif self._state.getAction() == 'down':
            if angle_X > 15:
                cv2.putText(frame, "down", (self._x, self._y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2) 
                return True
            else:
                cv2.putText(frame, "nothing", (self._x, self._y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)   
                return False
        elif self._state.getAction() == 'right': 
            if angle_Y > 15:
                cv2.putText(frame, "right", (self._x, self._y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)  
                return True
            else:
                cv2.putText(frame, "nothing", (self._x, self._y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)  
                return False
        elif self._state.getAction() == 'left':
            if angle_Y < -15:
                cv2.putText(frame, "left", (self._x, self._y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)   
                return True
            else:
                cv2.putText(frame, "nothing", (self._x, self._y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)   
                return False      



    def checkDirection(self, angle_X, angle_Y, angle_Z):
        # up, down, right, left'in dogru yapilip yapilmadigindan sorumlu olan metoddur.
        validness = self.is_looking_right_place((angle_X, angle_Y, angle_Z), frame)
        if validness == True: # kullanici istenen yone bakmaya basladiktan sonra True olur
            self._startTimer() # True olduktan sonra zamanlayici baslatilir
            if (self._start != None) and ((time.time() - self._start) > self._state.getOccurrenceNum()): # eger zamanlayici baslatilmissa ve baslatilmasinin ardindan gecen sure verilen sureyi gecmisse
                self._changeState() # ilgili action tamamlanmistir, state'i degistir
            if self._success_state_counter >= self._action_num: # eger tum action'lar tamamlanmissa
                cv2.putText(frame, "ACCESS GRANTED", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), thickness=2)
                print('ACCESS GRANTED')
                self._access_granted = True # bu degiskene True atanir ve program sonlanir
        else: # kullanici istenen yone su an bakmiyorsa
            if self._start != None: # ancak onceden bakmissa(yani zamanlayici baslamissa)
                self._validness_counter += 1 # frame basina yukarida da surekli bahsedilen degisken 1 artirilir.
        if self._validness_counter == self._validness_threshold: # eger bahsedilen esik degerini gecerse, yani belli bir sure istenen hareketi yapmazsa
            cv2.putText(frame, "ACCESS DENIED", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
            self._zeroCounter() # hareket reddedilir, gerekli degiskenler sifirlanir ve yeni action'lar yapilmasi icin atanir.



    def checkEmotion(self, face_img):
        # happy, neutral, etc.'nin dogru yapilip yapilmadigindan sorumlu olan metoddur.
        validness = self._emotion_detector.predict_emotion(face_img)
        if validness == self._state.getAction(): # kullanici istenen emotion durumuna girdikten sonra True olur
            cv2.putText(frame, self._state.getAction(), (self._x, self._y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)
            self._startTimer() # True olduktan sonra zamanlayici baslatilir
            if (self._start != None) and ((time.time() - self._start) > self._state.getOccurrenceNum()): # eger zamanlayici baslatilmissa ve baslatilmasinin ardindan gecen sure verilen sureyi gecmisse
                self._changeState() # ilgili action tamamlanmistir, state'i degistir
            if self._success_state_counter >= self._action_num:  # eger tum action'lar tamamlanmissa
                cv2.putText(frame, "ACCESS GRANTED", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), thickness=2)
                print('ACCESS GRANTED')
                self._access_granted = True # bu degiskene True atanir ve program sonlanir
        else: # kullanici o an istenen duygu durumunda degilse
            cv2.putText(frame, 'not ' + self._state.getAction(), (self._x, self._y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)
            if self._start != None: # ancak onceden istenen duygu durumunda bulunmussa(yani zamanlayici baslamissa)
                self._validness_counter += 1 # frame basina yukarida da surekli bahsedilen degisken 1 artirilir.
        if self._validness_counter == self._validness_threshold: # eger bahsedilen esik degerini gecerse, yani belli bir sure istenen duygu durumunda bulunmzasa
            cv2.putText(frame, "ACCESS DENIED", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
            self._zeroCounter() # hareket reddedilir, gerekli degiskenler sifirlanir ve yeni action'lar yapilmasi icin atanir.



    def checkBlinking(self, angle_X, angle_Y, angle_Z):
        cv2.putText(frame, str(self._eye_blink.getEyeBlinkNum()), (self._x, self._y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)
        self._startTimer() # goz kirpmada kontrol edilen sey self._blinking_sec suresinde yeteri miktarda goz kirpilip kirpilmadigi oldugu icin zamanlayici direkt baslatilir.
        self._face_stabler.warningLog(frame, (angle_X, angle_Y, angle_Z)) # yuz sabitleyici, suratin uygun pozisyonda olmamasi durumunda kullaniciyi yonlendirir(dogru pozisyona gelmesi icin)
        self._eye_blink.isBlinked(self._face_stabler) # goz kirpildiginda kendi icindeki degiskeni 1 artirir.
        # eger yeteri miktarda goz kirpilmissa ve hala verilen sure(self._blinking_sec) dolmadiysa
        if (self._eye_blink.getEyeBlinkNum() == self._state.getOccurrenceNum()) and ((time.time() - self._start) < self._blinking_sec): 
            self._changeState() # ilgili action tamamlanmistir, state'i degistir
        elif (time.time() - self._start) > self._blinking_sec: # eger verilen sure(self._blinking_sec) dolduysa
            cv2.putText(frame, "ACCESS DENIED", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
            self._zeroCounter() # hareket reddedilir, gerekli degiskenler sifirlanir ve yeni action'lar yapilmasi icin atanir.
        if self._success_state_counter >= self._action_num: # eger tum action'lar tamamlanmissa
            cv2.putText(frame, "ACCESS GRANTED", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), thickness=2)
            self._access_granted = True # bu degiskene True atanir ve program sonlanir



    def executer(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = LivenessDetection.detector(gray, 0)
        if len(rects) != 1:
            cv2.putText(frame, "THE SYSTEM GETS ACTIVATED WITH ONE PERSON", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)
            self._zeroCounter()
        else: # kamerada bir yuz varsa
            rect = rects[0] # ilgili yuz rect degiskenine atanir
            shape = LivenessDetection.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            angle_X, angle_Y, angle_Z = get_head_pose(shape) # 3 eksende kafa pozu alinir
            landmark_detection = LandmarkDetection(shape) # landmarklar cikarilir
            self._eye_blink.setLandmarkDetection(landmark_detection) # her frame'de guncellenir
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            (self._x, self._y) = (x, y)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if self._state.getLabel() == 'direction':
                self.checkDirection(angle_X, angle_Y, angle_Z)
            elif self._state.getLabel() == 'emotion':
                self.checkEmotion(frame[y:y+h,x:x+w])
            elif self._state.getLabel() == 'blinking':
                self.checkBlinking(angle_X, angle_Y, angle_Z)
            if self._state.getDone():
                cv2.putText(frame, self._state.getAction() + ' ' + str(self._state.getOccurrenceNum()), (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness=2)
            else:
                cv2.putText(frame, self._state.getAction() + ' ' + str(self._state.getOccurrenceNum()), (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)
            cv2.putText(frame, "X: " + "{:7.2f}".format(angle_X), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
            cv2.putText(frame, "Y: " + "{:7.2f}".format(angle_Y), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
            cv2.putText(frame, "Z: " + "{:7.2f}".format(angle_Z), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
            write_actions(self._actions_to_do)



cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to connect to camera.")

ld = LivenessDetection()

while ld.getAccessGranted() == False:
    ret, frame = cap.read()
    ld.executer(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   

if ld.getAccessGranted():
    t = Thread(target=sound_alarm, args=("access_granted.mp3",))   
    t.deamon = True
    t.start()   


cap.release()
cv2.destroyAllWindows()   
