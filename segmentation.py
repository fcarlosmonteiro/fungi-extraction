import numpy as np
import cv2

imagem = cv2.imread('i.png')
original = imagem.copy()
imagem1 = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

imagem = cv2.blur(imagem1,(15,15))

lower = np.array([0,37,0], dtype="uint8")
upper = np.array([86,255,193], dtype="uint8")
mascara = cv2.inRange(imagem, lower, upper)

cnts = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)


cv2.imshow('mascara', mascara)
cv2.imshow('original', original)
cv2.waitKey()