# Predict YOLO + UNet

Este script `predict_yoloUNet.py` procesa un video para realizar segmentación utilizando un modelo YOLO para definir la Región de Interés (ROI) y un modelo UNet para realizar la segmentación precisa, luego este resutlado es filtrado con ROI de YOLO, para elimianr la posibles segmentaciones fuera de lugar que puede producir UNet. El resultado es un video con los contornos de la glotis.
## Esrtuctura del repositorio

├── utils/
│   ├── models/
│   │   ├── best_yolov8n-seg-1cls.pt  # Modelo preentrenado YOLOv8 para segmentación
│   │   ├── epoch025.h5               # Modelo preentrenado UNet para segmentación
│   ├── __init__.py                   
│   ├── data.py                       # Funciones relacionadas con la carga y manejo de datos
│   ├── masks.py                      # Funciones para obtener y procesar máscaras de segmentación
│   ├── metrics.py                    # Cálculo de métricas
├── predict_yoloUNet.py               # Script principal para procesar videos
├── requirements.txt                  # Lista de Requerimientos

### Requerimientos 
opencv-python==4.10.0.82
imageio==2.34.2
tensorflow==2.15.1
tensorflow-addons==0.23.0
ultralytics==8.3.15
numpy==1.26.4
pandas==2.2.2
tqdm==4.66.4

## Comando de Ejecución

Ejecuta el script con el siguiente comando en la terminal:

```bash
python predict_yoloUNet.py --video '/ruta/video.avi' --output '/ruta/salida.mp4' --yolo_model '/ruta/yolo_model.pt' --unet_model '/ruta/unet_model.h5'
