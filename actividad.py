# Libreria openCV
import cv2
# Libreria NumPy
import numpy as np
#Liberia OS
import os

### Función que regresa una lista ordenada alfabéticamente de archivos con la extensión especificada en el directorio especificado
def getTestPicturesInPath(path, extension):
    pictures = []
    for file in os.listdir(path):
        if file.endswith(extension):
            pictures.append(os.path.join(path, file))
    
    pictures.sort()
    return pictures


# Obtiene e imprime la lista de imagenes jpeg dentro del directorio pictures/
picturePathList = getTestPicturesInPath("pictures/", ".jpeg")
print("Imagenes a procesar: " + str(len(picturePathList)))

# Ciclo para procesar cada una de las imagenes
for picturePath in picturePathList:
    # Leemos la imagen con sus colores originales
    src = cv2.imread(picturePath, cv2.IMREAD_UNCHANGED)

    cv2.imshow(picturePath,src)

    # Extraer el color verde
    # The HSV color space converted from BGR
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    # Define the blue range in HSV color space
    lower_green = np.array([40,40,40])
    upper_green = np.array([70,255,255])    

    # The masking threshold generator #
    greenMask = cv2.inRange(hsv, lower_green, upper_green)
    cv2.imshow('green mask',greenMask)
    # Fin extraccion verde

    # Erosion
    # Creating kernel
    structuralElementSize = 5
    kernel = np.ones((structuralElementSize, structuralElementSize), np.uint8)
    # Using cv2.erode() method 
    img_erosion = cv2.erode(greenMask, kernel, iterations=1)
    cv2.imshow('Imagen erosionada',img_erosion)     

    # Fin erosion

    # Dilatacion -  necesaria?
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    cv2.imshow('Imagen dilatada',img_dilation)     
    # Fin dilatacion

    

    cv2.waitKey(0) 
    cv2.destroyAllWindows()