import numpy as np
import cv2

def rescale(src):
    #percent by which the image is resized
    scale_percent = 30

    #calculate the 50 percent of original dimensions
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(src, dsize)
    
    return output

i = cv2.imread('i4.jpg')
imagem=rescale(i)

original = imagem.copy()
imagem1 = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
#cv2.imshow("color",imagem1)

imagem = cv2.GaussianBlur(imagem1,(95,95),cv2.BORDER_DEFAULT)

#cv2.imshow("Guassian2",imagem)

lower = np.array([0,30,0], dtype="uint8")
upper = np.array([46,255,255], dtype="uint8")
mascara = cv2.inRange(imagem, lower, upper)

cnts = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

#for c in cnts:
#seleciona o contorno maior
c = max(cnts, key = cv2.contourArea)

#seleciona as coordenadas
x,y,w,h = cv2.boundingRect(c)
print([x,y,w,h])

#plota na imagem
cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
cv2.imshow('original', original)

cv2.rectangle(mascara, (x, y), (x + w, y + h), (36,255,12), 2)
cv2.imshow('mascara', mascara)

#calcula o moments somente para o contorno maior
M = cv2.moments(c)
print(M)


cv2.waitKey()