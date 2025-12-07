import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
#X (p x N)

#Y (C X N)

#C = 20
#N = 32 * 20 = 640
#p = 15

lista_pessoas = os.listdir('RecFac')
C = len(lista_pessoas)
R = 200

X = np.empty((R*R,0))
Y = np.empty((C,0))
i = 0
for pessoa in lista_pessoas:   
    lista_imagens = os.listdir(f'RecFac\\{pessoa}')

    rotulo = -np.ones((C,1))
    rotulo[i,0] = 1
    Y = np.hstack((
        Y, np.tile(rotulo, (1,len(lista_imagens)))
    ))
    
    i+=1
    for imagem in lista_imagens:
        img = cv2.imread(f"RecFac\\{pessoa}\\{imagem}", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (R,R))
        x = img.flatten()
        X = np.hstack((
            X, x.reshape(R*R,1)
        ))
        cv2.imshow(f"{pessoa}", img)
        cv2.waitKey(0)
