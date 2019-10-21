import numpy as np


from matplotlib import pyplot as plt

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2


if len(sys.argv) != 2:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)
if sys.argv[1].lower() == "orb":
  detector = 1
elif sys.argv[1].lower() == "kaze":
  detector = 2
else:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)



#Lecture de la paire d'images

img = cv2.imread('../Image_Pairs/torb_small1.png')
img1 = cv2.imread('../Image_Pairs/torb_small1.png')
img2 = cv2.imread('../Image_Pairs/torb_small1.png')
print("Dimension de l'image 1 :",img1.shape[0],"lignes x",img1.shape[1],"colonnes")
print("Type de l'image 1 :",img1.dtype)

#Distortion of the image
rows,cols,ch = img.shape
#gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
pt= (cols/2, rows/2)
r=cv2.getRotationMatrix2D(pt, 10, 1.0 )
imgDistort=cv2.warpAffine(img, r,(cols,rows))
plt.imshow(imgDistort, cmap = 'gray')
print(ch)
plt.show()
gray =  imgDistort

#Début du calcul
t1 = cv2.getTickCount()
#Création des objets "keypoints"
if detector == 1:
  kp1 = cv2.ORB_create(nfeatures = 250,#Par défaut : 500
                       scaleFactor = 2,#Par défaut : 1.2
                       nlevels = 3)#Par défaut : 8
  kp2 = cv2.ORB_create(nfeatures=250,
                        scaleFactor = 2,
                        nlevels = 3)

  kp3 = cv2.ORB_create(nfeatures=250,
                        scaleFactor = 2,
                        nlevels = 3)

  print("Détecteur : ORB")
else:
  kp1 = cv2.KAZE_create(upright = False,#Par défaut : false
    		        threshold = 0.001,#Par défaut : 0.001
  		        nOctaves = 4,#Par défaut : 4
		        nOctaveLayers = 4,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  kp2 = cv2.KAZE_create(upright = False,#Par défaut : false
	  	        threshold = 0.001,#Par défaut : 0.001
		        nOctaves = 4,#Par défaut : 4
		        nOctaveLayers = 4,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  kp3 = cv2.KAZE_create(upright = False,#Par défaut : false
	  	        threshold = 0.001,#Par défaut : 0.001
		        nOctaves = 4,#Par défaut : 4
		        nOctaveLayers = 4,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  print("Détecteur : KAZE")
#Conversion en niveau de gris
gray1 =  cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray =  cv2.cvtColor(imgDistort,cv2.COLOR_BGR2GRAY)
gray2 =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#Détection des keypoints
pts3,desc3 = kp3.detectAndCompute(gray2,None)
pts1, desc1 = kp1.detectAndCompute(gray1,None)
pts, desc = kp2.detectAndCompute(gray,None)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Détection des points d'intérêt :",time,"s")

#Affichage des keypoints
img1 = cv2.drawKeypoints(gray1, pts1, None, flags=4)
# flags définit le niveau d'information sur les points d'intérêt
# 0 : position seule ; 4 : position + échelle + direction
img = cv2.drawKeypoints(gray, pts, None, flags=4)
img2= cv2.drawKeypoints(gray2, pts3, None, flags=4)


matches = flann.knnMatch(desc1,desc,k=2)
matches1 = flann.knnMatch(desc1,desc3,k=2)
good = []
for m,n in matches:
  if m.distance < 0.7*n.distance:
    good.append([m])

good1 = []
for m,n in matches1:
  if m.distance < 0.7*n.distance:
    good1.append([m])

print(len(good/good1))
plt.subplot(121)
plt.imshow(img1)
plt.title('Image n°1')

plt.subplot(122)
plt.imshow(img)
plt.title('Image n°2')

plt.show()

plt.subplot(121)
plt.imshow(img1)
plt.title('Image n°1')

plt.subplot(122)
plt.imshow(img2)
plt.title('Image n°2')

plt.show()


rows,cols,ch = img1.shape
#gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
pt= (cols/2, rows/2)
r=cv2.getRotationMatrix2D(pt, 10, 1.0 )
imgDistort1=cv2.warpAffine(img1, r,(cols,rows))
plt.imshow(imgDistort1, cmap = 'gray')
plt.show()

plt.subplot(121)
plt.imshow(img)
plt.title('Image n°1')

plt.subplot(122)
plt.imshow(imgDistort1)
plt.title('Image n°2')
plt.show()

img4= cv2.drawMatches(img,kp1,img1,kp2,matches1, flags=2)
img5= cv2.drawMatches(img1,kp2,img2,kp3, matches2, flags=2)

plt.subplot(121)
plt.imshow(img4)
plt.title('Image n°1')

plt.subplot(122)
plt.imshow(img5)
plt.title('Image n°2')
plt.show()
