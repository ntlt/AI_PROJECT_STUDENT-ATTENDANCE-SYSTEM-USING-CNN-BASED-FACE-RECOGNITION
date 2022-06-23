import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from tensorflow import keras
from datetime import datetime

facedetect = cv2.CascadeClassifier("/home/trang/Downloads/Face Recognition System (1)-20220622T194744Z-001/Face Recognition System (1)/Face Recognition System/haarcascade_frontalface_default.xml")


cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font=cv2.FONT_HERSHEY_COMPLEX

model = load_model("/home/trang/Downloads/Face Recognition System (1)-20220622T194744Z-001/Face Recognition System (1)/Face Recognition System/Face.h5")

def get_className(classNo):
	if classNo==0:
		return "Gia Han"
	elif classNo==1:
		return "Linh Trang"
	elif classNo==2:
		return "Ngoc Mai"
	elif classNo==3:
		return "Nhu Quynh"
	elif classNo==4:
		return "Ha"
	elif classNo==5:
		return "Truc Anh"
	elif classNo==6:
		return "Van Minh"


		
while True:
	sucess, imgOrignal=cap.read()
	faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
	for x,y,w,h in faces:
		crop_img=imgOrignal[y:y+h,x:x+h]
		img=cv2.resize(crop_img, (100,100))
		img=img.reshape(1, 100, 100, 3)
		prediction=model.predict(img)
		classIndex=model.predict(img)
		classes_x=np.argmax(prediction,axis=1)
		probabilityValue=np.amax(prediction)
		if classes_x==0:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
			cv2.putText(imgOrignal, str(get_className(classes_x)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			print(str(get_className(classes_x)))
		elif classes_x==1:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
			cv2.putText(imgOrignal, str(get_className(classes_x)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
			print(str(get_className(classes_x)))
		
		name = str(get_className(classes_x))

		def markAttendance(name):
			with open('/home/trang/Downloads/Face Recognition System (1)-20220622T194744Z-001/Face Recognition System (1)/Face Recognition System/Diem_danh.csv','r+') as f:
				myDataList = f.readlines()
				nameList = []
				for line in myDataList:
					entry = line.split(',')
					nameList.append(entry[0])
				if name not in nameList:
					now = datetime.now()
					dtString = now.strftime('%H:%M:%S')
					f.writelines(f'\n{name},{dtString}')

		cv2.putText(imgOrignal,str(round(probabilityValue*100, 2))+"%" ,(180, 75), font, 0.75, (255,0,0),2, cv2.LINE_AA)
		markAttendance(name)
		
	cv2.imshow("Result",imgOrignal)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break


cap.release()
cv2.destroyAllWindows()





















