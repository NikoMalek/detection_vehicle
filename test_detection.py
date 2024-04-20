import argparse
from typing import List
import numpy as np
import cv2
import numpy as np
from inference import get_model
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer
import supervision as sv
import sqlite3
from datetime import datetime, timedelta
from utils.general import get_stream_frames_generator
# Database

conn = sqlite3.connect('vehicle_detections.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS vehicle_detections
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              vehicle_type TEXT,
              detection_time TEXT,
              time_in_zone REAL)''')


COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
    
)


def main(
    source_video_path: str,
    zone_configuration_path: str,
    model_id: str,
    confidence: float,
    iou: float,
    classes: List[int],
) -> None:
    with open('Clases.txt', 'r') as f:
        class_mapping = {int(line.split(': ')[0]): line.split(': ')[1].strip() for line in f.readlines() if line.strip()}
    model = get_model(model_id=model_id)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
       # Si es un archivo de video local, usa sv.get_video_frames_generator()
    if source_video_path.startswith('http') or source_video_path.startswith('rtsp'):
        cap = cv2.VideoCapture(source_video_path)
        if not cap.isOpened():
            print("Error al abrir el flujo RTSP")
            return
        video_info = sv.VideoInfo(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                 height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                 fps=cap.get(cv2.CAP_PROP_FPS))
        frames_generator = get_stream_frames_generator(source_video_path)
        print(f"FPS de stream: {video_info.fps} fps")
    else:
        video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
        frames_generator = sv.get_video_frames_generator(source_video_path)
    
    # Limitar la tasa de fotogramas a 30 FPS para evitar errores de cálculo de segundos
    if video_info.fps > 30:
        video_info.fps = 30
        print(f"FPS limitado: {video_info.fps} fps")
    resolution_wh = video_info.width, video_info.height
    polygons = load_zones_config(file_path=zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=resolution_wh,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    timers = [FPSBasedTimer(video_info.fps) for _ in zones]
    tracked_objects = {}

    for frame in frames_generator:
        results = model.infer(frame, confidence=confidence, iou_threshold=iou)[0]
        detections = sv.Detections.from_inference(results)
        detections = detections[find_in_list(detections.class_id, classes)]
        detections = tracker.update_with_detections(detections)
        annotated_frame = frame.copy()

        for idx, zone in enumerate(zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            )

            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = timers[idx].tick(detections_in_zone)
            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)
            annotated_frame = COLOR_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup,
            )
            labels = [
                f"#{tracker_id} {class_mapping[class_id]} {int(time)}"
                for tracker_id, class_id, time in zip(detections_in_zone.tracker_id, detections_in_zone.class_id, time_in_zone)
            ]
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup,
            )

            for tracker_id, class_id, time in zip(detections_in_zone.tracker_id, detections_in_zone.class_id, time_in_zone):
                if tracker_id not in tracked_objects:
                    tracked_objects[tracker_id] = {
                        'vehicle_type': class_mapping[class_id],
                        'detection_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'time_in_zone': round(timedelta(seconds=time).total_seconds())
                    }
                else:
                    tracked_objects[tracker_id]['time_in_zone'] = round(timedelta(seconds=time).total_seconds())

        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    for object_id, object_data in tracked_objects.items():
        print(f"Vehicle type: {object_data['vehicle_type']}, Detection time: {object_data['detection_time']}, Time in zone: {object_data['time_in_zone']}")
        if object_data['time_in_zone'] >= 1:
            c.execute("INSERT INTO vehicle_detections (vehicle_type, detection_time, time_in_zone) VALUES (?, ?, ?)", (object_data['vehicle_type'], object_data['detection_time'], object_data['time_in_zone']))
    conn.commit()
    cv2.destroyAllWindows()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculating detections dwell time in zones, using video file."
    )
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        required=True,
        help="Path to the zone configuration JSON file.",
    )
    parser.add_argument(
        "--source_video_path",
        type=str,
        required=True,
        help="Path to the source video file.",
    )
    parser.add_argument(
        "--model_id", type=str, default="yolov8s-640", help="Roboflow model ID."
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Confidence level for detections (0 to 1). Default is 0.3.",
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        type=float,
        help="IOU threshold for non-max suppression. Default is 0.7.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=[],
        help="List of class IDs to track. If empty, all classes are tracked.",
    )
    args = parser.parse_args()

    main(
        source_video_path=args.source_video_path,
        zone_configuration_path=args.zone_configuration_path,
        model_id=args.model_id,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        classes=args.classes,
    )
    print("Done! ")