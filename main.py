import cv2
import numpy as np

# Ouvrez la vidéo en spécifiant le chemin du fichier
video_capture = cv2.VideoCapture('chemin/du/fichier/video.mp4')

# Chargez le classifieur de visage en spécifiant le chemin du fichier
face_classifier = cv2.CascadeClassifier('chemin/du/fichier/classifieur.xml')

while True:
    # Lisez le prochain frame
    ret, frame = video_capture.read()

    # Si le frame est None, cela signifie que la vidéo est terminée
    if frame is None:
        break

    # Convertir l'image en niveaux de gris pour une détection de visage plus efficace
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages dans l'image
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # Pour chaque visage détecté, créez un masque de flou et utilisez-le pour flouter le visage dans l'image
    for (x,y,w,h) in faces:
        # Créer un masque de flou en utilisant une fonction Gaussienne
        blur_mask = cv2.GaussianBlur(frame[y:y+h, x:x+w], (99,99), 30)

        # Appliquer le masque de flou sur l'image
        frame[y:y+h, x:x+w] = blur_mask

    # Afficher l'image floutée
    cv2.imshow('Video', frame)

    # Attendre que l'utilisateur appuie sur la touche 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture de vidéo et détruire les fenêtres ouvertes
video_capture.release()
cv2.destroyAllWindows()