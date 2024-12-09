import numpy as np
import cv2

def get_unet_mask(image, unet_model):
    img_orig = image.astype(np.float32)

    # Convertir a gris y normalizar la imagen
    img_orig_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img_orig_gray, (256, 512))

    # Normalizar y preprocesar la imagen
    normalizedImg = np.zeros(img.shape)
    img = cv2.normalize(img, normalizedImg, -1, 1, cv2.NORM_MINMAX)
    img = img[None, ..., None]

    # Predicción
    pred_maps, seg_pred =unet_model.predict(img)
    mask = np.asarray(np.squeeze(seg_pred))

    # Convertir las probabilidades a booleanas
    mask = np.round(mask)
    # Redimensionar la máscara al tamaño original del frame
    mask = cv2.resize(mask, (img_orig_gray.shape[1], img_orig_gray.shape[0]))
    mask = mask.astype(bool)
    return mask

def get_max_yolo_roi(video_path, yolo_model, margin=15):
    """
    Obtiene la región de interés (ROI) que cubre todas las detecciones de YOLO en un video completo,
    imprime todas las cajas detectadas y muestra la imagen con la ROI máxima que cubre todas las detecciones.

    Parámetros:
    video_path (str): Ruta del video.
    yolo_model (YOLO): Modelo YOLO preentrenado.
    margin (int): Tamaño del margen a agregar alrededor de la ROI.

    Retorna:
    tuple: Coordenadas (x1, y1, x2, y2) de la ROI máxima detectada con márgenes aplicados.
    """
    cap = cv2.VideoCapture(video_path)
    min_x1, min_y1 = float('inf'), float('inf')
    max_x2, max_y2 = 0, 0
    max_img = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().int().numpy()

                # Asegurarse de que las coordenadas están dentro del tamaño de la imagen
                x1 = max(0, min(x1, frame.shape[1] - 1))
                y1 = max(0, min(y1, frame.shape[0] - 1))
                x2 = max(0, min(x2, frame.shape[1] - 1))
                y2 = max(0, min(y2, frame.shape[0] - 1))

                # Actualizar los valores mínimos y máximos para obtener una ROI que englobe todas las detecciones
                min_x1 = min(min_x1, x1)
                min_y1 = min(min_y1, y1)
                max_x2 = max(max_x2, x2)
                max_y2 = max(max_y2, y2)

                max_img = frame.copy()

    cap.release()

    # Agregar márgenes a la ROI que engloba todas las detecciones
    if max_img is not None:

        # Aplicar márgenes asegurándose de que no excedan los límites de la imagen
        min_x1 = max(0, min_x1 - margin)
        min_y1 = max(0, min_y1 - margin)
        max_x2 = min(max_img.shape[1] - 1, max_x2 + margin)
        max_y2 = min(max_img.shape[0] - 1, max_y2 + margin)

        max_roi_with_margin = (min_x1, min_y1, max_x2, max_y2)

    else:
        print("No se detectó ninguna ROI en las imágenes.")

    return max_roi_with_margin


def filter_unet_mask_with_yolo(unet_mask, roi):
    """
    Filtra la máscara de UNet para mantener solo las áreas dentro de la ROI proporcionada por YOLO.

    Parámetros:
    unet_mask (numpy.ndarray): Máscara binaria de UNet.
    roi (tuple): Coordenadas de la ROI (x1, y1, x2, y2).

    Retorna:
    numpy.ndarray: Máscara filtrada que mantiene solo las áreas dentro de la ROI.
    """
    x1, y1, x2, y2 = roi
    filtered_mask = np.zeros_like(unet_mask)
    filtered_mask[y1:y2, x1:x2] = unet_mask[y1:y2, x1:x2]
    return filtered_mask
