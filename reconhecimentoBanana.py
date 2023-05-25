import cv2

cascade = cv2.CascadeClassifier('banana_classifier.xml')

imagem = cv2.imread('images/Banana.jpg')

imgGray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

faces = cascade.detectMultiScale(imgGray, scaleFactor=1.08, minNeighbors=4)

for(x,y,w,h) in faces:
    dectada = cv2.rectangle(imagem,(x,y),(x+w,y+h), (255,0,0),2)

cv2.imshow('imagem', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

