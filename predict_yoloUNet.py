import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import imageio as io
import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import gc
import argparse
from utils.masks import get_unet_mask, get_max_yolo_roi, filter_unet_mask_with_yolo
from segmentation_models.losses import dice_loss
from segmentation_models.metrics import iou_score
from utils.data import load_data, metric_mape, mape_ap, mape_pp


def process_and_create_segmented_video_with_contours(video_path, yolo_model, unet_model, output_video_path, margin=10):
    video_cap = cv2.VideoCapture(video_path)

    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Calculo de ROI utilizando YOLO
    roi = get_max_yolo_roi(video_path, yolo_model, margin)

    frame_count = 0
    while video_cap.isOpened():
        ret, frame_video = video_cap.read()

        if not ret:
            break

        # Obtener la máscara de UNet
        unet_mask = get_unet_mask(frame_video, unet_model)

        # Filtrar la máscara de UNet con la ROI de YOLO
        filtered_mask = filter_unet_mask_with_yolo(unet_mask, roi)

        filtered_mask_resized = cv2.resize(filtered_mask.astype(np.uint8), (frame_width, frame_height))

        contours, _ = cv2.findContours(filtered_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_with_contours = frame_video.copy()  # Crear una copia del frame original
        cv2.drawContours(frame_with_contours, contours, -1, (0, 255, 0), 2)  # Dibujar contornos en verde

        output_video.write(frame_with_contours)

        frame_count += 1

        if frame_count % 100 == 0:
            gc.collect()
            tf.keras.backend.clear_session()

    video_cap.release()
    output_video.release()

    print(f"Video con segmentación y contornos creado en: {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description="Procesa un video con segmentación y contornos.")
    parser.add_argument("--video", required=True, help="Ruta al video de entrada.")
    parser.add_argument("--output", required=True, help="Ruta al video de salida.")
    parser.add_argument("--yolo_model", required=True, help="Ruta al modelo YOLO.")
    parser.add_argument("--unet_model", required=True, help="Ruta al modelo UNet.")

    args = parser.parse_args()

    # Cargar modelos
    yolo_model = YOLO(args.yolo_model)
    unet_model = load_model(args.unet_model, custom_objects={"tfa": tfa, "dice_loss": dice_loss, "metric_mape": metric_mape, "iou_score": iou_score, "mape_ap": mape_ap, "mape_pp": mape_pp})

    # Procesar el video
    process_and_create_segmented_video_with_contours(args.video, yolo_model, unet_model, args.output)

if __name__ == "__main__":
    main()
