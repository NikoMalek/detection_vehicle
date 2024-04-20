# detection_vehicle


## 💻 Instalación

- Clonar el repositorio

  ```bash
  git clone https://github.com/NikoMalek/detection_vehicle.git
  ```

- Configurar el entorno de Python 3.10.14 con conda y activarlo. [Descarga de conda](https://docs.anaconda.com/free/miniconda/index.html)
  ```bash
  conda create --name mi_entorno python=3.10.14
  conda activate mi_entorno
  ```
- Instalación de las dependencias necesarias desde el archivo requirements.txt
  ```bash
  pip install -r requirements.txt
  ```
- Para desactivar el entorno virtual al terminar:
  ```bash
  conda deactivate
  ```
  ### Uso de GPU
  Se recomienda el uso de GPU con CUDA cores ([GPU compatibles](https://developer.nvidia.com/cuda-gpus)) para un mejor rendimiento.
  Se requiere tener preinstalado [CUDA 12.x](https://developer.nvidia.com/cuda-downloads) y [cuDNN](https://developer.nvidia.com/cudnn-downloads) 
- Para utilizar la GPU primero desinstalamos onnxruntime-gpu e instalamos la versión necesaria
  ```bash
  pip uninstall onnxruntime-gpu
  ```
  ```bash
  pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
  ```
- Instalamos la versión de PyTorch requerida para la utilización de CUDA
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

  
## 🛠 Script de dibujo de zona de interés

### `draw_zones.py`

Este script permite personalizar las zonas de detección. Puedes dibujar las zonas directamente sobre un video o stream y los resultados se guardarán en un archivo JSON. Al ejecutar el script, se abrirá una ventana donde podrás dibujar polígonos en el primer frame de la fuente (video o stream). Las coordenadas de los polígonos se almacenarán en un archivo JSON.

- `--source_path`: Directorio de la fuente del video o stream para dibujar los polígonos.

- `--zone_configuration_path`: Directorio en donde se guardaran las coordenadas de los polígonos dibujados en un archivo JSON.


Uso de la ventana para dibujar polígonos

- **clic izquierdo** - establecer la posición de los vértices.

- **enter** - terminar de dibujar el polígono actual.

- **escape** - cancelar el dibujo del polígono actual.

- **q** - cerrar la ventana de dibujado sin guardar.

- **s** - guardar las zonas dibujadas en un archivo JSON.

Dibujo de polígonos en archivo de video local:

```bash
python draw_zones.py --source_path "videos/videocut.mp4" --zone_configuration_path "data/traffic/config.json"
```

Dibujo de polígonos en stream de video en vivo:
- `http`
```bash
python draw_zones.py --source_path "http://" --zone_configuration_path "data/traffic/configstream.json"
```
- `rtsp`
```bash
python draw_zones.py --source_path "rtsp://" --zone_configuration_path "data/traffic/configstream.json"
```

### Ejemplo de uso:


https://github.com/NikoMalek/detection_vehicle/assets/114967158/6b60d822-7614-45b0-88b7-1aafbff14f8a




##  🎬 Iniciar detección:

### `test_detection.py`
Este script permite la detección de objetos en un archivo de video local o stream mediante protocolo http o rtsp.

- `--zone_configuration_path`: Especifica la ruta al archivo de configuración de las zonas, donde se definen los polígonos que representan las áreas de interés en el video.

- `--source_video_path`: Establece la ruta al archivo de video que se utilizará como entrada para el procesamiento(Archivos de video local en formato MP4 y streams de video en vivo en formato http o rtsp)

- `--model_id`: Establece el ID del modelo de detección de objetos que se utilizará. El valor predeterminado es "yolov8s-640". Las opciones ordenadas de más ligera a más pesada son: (YOLOv8n - YOLOv8s - YOLOv8m - YOLOv8l - YOLOv8x)

- `--classes`: Permite especificar una lista de IDs de clases que se deben detectar(Clases en Clases.txt)

- `--confidence_threshold`: Establece el umbral de confianza mínimo para las detecciones de objetos. Sólo se tendrán en cuenta las detecciones con un nivel de confianza superior a este valor. El valor predeterminado es 0.3.

- `--iou_threshold`: Establece el umbral de Intersection over Union (IoU) para la supresión de no máximos. Este parámetro se utiliza para eliminar detecciones duplicadas. El valor predeterminado es 0.7.

Detección de archivo de video local: 
```bash
python test_detection.py --zone_configuration_path "data/traffic/config.json" --source_video_path "videos/videocut.mp4" --model_id "yolov8l-640" --classes 2 3 5 7 --confidence_threshold 0.3 --iou_threshold 0.7
```

Detección de stream de video en vivo:
- `http`
```bash
python test_detection.py --zone_configuration_path "data/traffic/configstream.json" --source_video_path "http://" --model_id "yolov8l-640" --classes 2 3 5 7 --confidence_threshold 0.3 --iou_threshold 0.7
```
- `rtsp`
```bash
python test_detection.py --zone_configuration_path "data/traffic/configstream.json" --source_video_path "rtsp://" --model_id "yolov8l-640" --classes 2 3 5 7 --confidence_threshold 0.3 --iou_threshold 0.7
```
### Ejemplo de uso con archivo de video local:


https://github.com/NikoMalek/detection_vehicle/assets/114967158/2c52b4e8-f02b-4892-acde-720bf1188127

- Se registran y clasifican en la base de datos SQLite un total de 21 vehículos que transitan por la zona de interés.

![base-de-datos](https://github.com/NikoMalek/detection_vehicle/assets/114967158/c3c6add0-c339-4ba4-8eef-fd59b72adfc8)



## © license

Este programa integra dos componentes principales, cada uno con su propia licencia:

- ultralytics: El modelo de detección utilizado en este programa, YOLOv8, se distribuye bajo la [licencia AGPL-3.0] 
  se distribuye bajo la [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
  Puede encontrar más detalles sobre esta licencia aquí.

- supervision: El código analítico que impulsa el análisis basado en zonas en esta demo se basa en la biblioteca
  se basa en la biblioteca Supervision, que se distribuye bajo la
  [MIT license](https://github.com/roboflow/supervision/blob/develop/LICENSE.md).  Esto
  hace que la parte de Supervision del código sea totalmente de código abierto y sé
  pueda reutilizar libremente en sus proyectos.
