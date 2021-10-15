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

i = cv2.imread('i1.jpg')
imagem=rescale(i)

#remove ruidos
image = cv2.fastNlMeansDenoisingColored(imagem,None,12,12,7,21)
#cv2.imshow("desnoise",image)

#colore imagem
ds = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
#cv2.imshow("color after desnoise",ds)

#aplica blur pra tirar mias imperfeicoes
img_hsv = cv2.GaussianBlur(ds,(95,95),cv2.BORDER_DEFAULT)

result = img_hsv.copy()
image = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2HSV)
lower = np.array([0,10,10])
upper = np.array([56,255,255])
mask = cv2.inRange(img_hsv, lower, upper)
result = cv2.bitwise_and(result, result, mask=mask)

#converte em cinza
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

#marca area de interesse
edge_detected_image = cv2.Canny(gray, 175, 200)
#cv2.imshow('Edge', edge_detected_image)

#contorna area de interesse
cnts = cv2.findContours(edge_detected_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

#seleciona o contorno maior
c = max(cnts, key = cv2.contourArea)

#seleciona as coordenadas
x,y,w,h = cv2.boundingRect(c)
print([x,y,w,h])

#plota na imagem
cv2.rectangle(imagem, (x, y), (x + w, y + h), (36,255,12), 2)
cv2.imshow('original', imagem)

cv2.rectangle(result, (x, y), (x + w, y + h), (36,255,12), 2)
cv2.imshow('result', result)

#calcula o moments somente para o contorno maior
M = cv2.moments(c)
print(M)

cv2.waitKey()