from typing import List, Optional, Union
import numpy as np
import torch
from collections import deque
import norfair
from norfair import Detection, Paths, Tracker, Video
from norfair.distances import frobenius, iou, mean_euclidean, mean_manhattan, iou_opt
import cv2
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple
if TYPE_CHECKING:
    from norfair.tracker import Detection, TrackedObject
from norfair.filter import OptimizedKalmanFilterFactory
from norfair import (
    AbsolutePaths,
    Detection,
    FixedCamera,
    Tracker,
    Video,
    draw_absolute_grid,
    draw_tracked_boxes,
)
from norfair.camera_motion import (
    HomographyTransformationGetter,
    MotionEstimator,
    TranslationTransformationGetter,
)
from threading import Thread, Lock

cv2.startWindowThread()

class WebcamVideoStream :
    def __init__(self, src = 0 , width = 1920, height = 1080) : # сначала разрешение настраивается тут
        self.stream = cv2.VideoCapture(src)   # возможность переключения на видео
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv2.getTickCount()
        self._freq = 1000.0 / cv2.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv2.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 0)

        return fps_rounded

cv_fps_calc = CvFpsCalc(buffer_len=10)

class YOLO:
    def __init__(self, model_name: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
           # load model
        #путь к yolo
        self.model = torch.hub.load(r'C:\Users\gleba\.cache\torch\hub\ultralytics_yolov5_master', 'custom', path=model_name, source='local')
        
    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None,
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections

def center(points):
    return [np.mean(np.array(points), axis=0)]

def center_draw_circle(points):
    centerx = points[0][0]+(points[1][0]-points[0][0])/2
    centery = points[1][1]+(points[0][1]-points[1][1])/2
    return (centerx, centery)

def draw_tracked_circle(frame: np.ndarray, objects: Sequence["TrackedObject"], draw_points: bool = True, draw_labels: bool = False, scale_factor_distance: int = 100, scale_factor_circle: int = 930):
    frame_scale = frame.shape[0] / 100
    for obj in objects:
        if not obj.live_points.any():
            continue
        points = obj.estimate
        if draw_points:
            points = points.astype(int)
            size = int((abs(points[1][0] - points[0][0]) * abs(points[0][1] - points[1][1]))/scale_factor_circle)
            cnt = center_draw_circle(points)
            cv2.circle(frame, (int(cnt[0]), int(cnt[1])), size, (0,255,255), 2)
            cv2.line(frame, (int(cnt[0]-size), int(cnt[1]+size)), (int(cnt[0]+size), int(cnt[1]-size)), (0,255,255), 2)
            cv2.line(frame, (int(cnt[0]+size), int(cnt[1]+size)), (int(cnt[0]-size), int(cnt[1]-size)), (0,255,255), 2)
            cv2.circle(frame, (int(cnt[0]), int(cnt[1])), 3, (0,255,255), 4)
        
        if draw_labels:
            label_draw_position = np.array(points[0, :])           

def yolo_detections_to_norfair_detections(yolo_detections: torch.tensor, track_points: str = "centroid") -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [detection_as_xywh[0].item(), detection_as_xywh[1].item()]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(
                    points=centroid,
                    scores=scores,
                    label=int(detection_as_xywh[-1].item()),
                )
            )
    elif track_points == "bbox":
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
                ]
            )
            scores = np.array(
                [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
            )
            norfair_detections.append(
                Detection(
                    points=bbox, scores=scores, label=int(detection_as_xyxy[-1].item())
                )
            )

    return norfair_detections

def yolo_detections_bbox_classes_zero(yolo_detections: torch.tensor) -> List[Detection]:
    norfair_detections: List[Detection] = []
    detections_as_xyxy = yolo_detections.xyxy[0]
    for detection_as_xyxy in detections_as_xyxy:
        if detection_as_xyxy[-1] == 0.:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
                ]
            )

            scores = np.array(
                [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
            )
            norfair_detections.append(
                Detection(points=bbox, scores=scores, label=int(detection_as_xyxy[-1].item()))
            )
    return norfair_detections

def yolo_detections_bbox_classes_one(yolo_detections: torch.tensor) -> List[Detection]:
    norfair_detections: List[Detection] = []
    detections_as_xyxy = yolo_detections.xyxy[0]
    for detection_as_xyxy in detections_as_xyxy:
        if detection_as_xyxy[-1] == 1.:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
                ]
            )
            scores = np.array(
                [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
            )
            norfair_detections.append(
                Detection(points=bbox, scores=scores, label=int(detection_as_xyxy[-1].item()))
            )
    return norfair_detections

def yolo_detections_centroid_classes_one(yolo_detections: torch.tensor) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []
    detections_as_xywh = yolo_detections.xywh[0]
    for detection_as_xywh in detections_as_xywh:
        if detection_as_xywh[-1] == 1.:
            centroid = np.array([detection_as_xywh[0].item(), detection_as_xywh[1].item()])
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(Detection(points=centroid, scores=scores, label=int(detection_as_xywh[-1].item())))
    return norfair_detections

def circle_frame(frame, radius):
    w = frame.shape[1]
    h = frame.shape[0]
    cv2.line(frame, (int(w/2), 0+500), (int(w/2), h-500), (255,255,255), thickness=2 )
    cv2.line(frame, (500, int(h/2)), (w-500, int(h/2)), (255,255,255), thickness=2 )
    #cv2.circle(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), radius, (0,0,255), 2)
    return frame

#датасет
model = YOLO(model_name='crowdhuman_yolov5m.pt', device='cpu')

capture = WebcamVideoStream().start()
DISTANCE_THRESHOLD_BBOX: float = 10
DISTANCE_THRESHOLD_CENTROID: int = 50
MAX_DISTANCE: int = 100
tracker_people = Tracker(distance_function=iou_opt, distance_threshold=50, hit_counter_max=10)
tracker_head = Tracker(distance_function=iou_opt, distance_threshold=50, hit_counter_max=10)
frame_counter = 0
detections = []
detections_head = []
detections_people = []
people_counter = 0 
skip_period = 1

while True:
    frame = capture.read()
    fps = cv_fps_calc.get()
    frame_counter+=1
    center_head_x, center_head_y = (0,0)
    
    if frame_counter % skip_period == 0:
        yolo_detections = model(frame, conf_threshold=0.7, iou_threshold = 0.3, image_size = 640, classes=[0, 1])
        detections_people = yolo_detections_bbox_classes_zero(yolo_detections)
        detections_head = yolo_detections_bbox_classes_one(yolo_detections)
    
    tracked_objects_people = tracker_people.update(detections=detections_people, period=skip_period)
    tracked_objects_head = tracker_head.update(detections=detections_head, period=skip_period)
    people_counter_cadr = len(tracked_objects_people)
    for obj in tracked_objects_people:
        people_counter=obj.id
    norfair.draw_tracked_boxes(frame, tracked_objects_people, id_size=0, id_thickness=2, border_colors= (0,255,0))
    draw_tracked_circle(frame, tracked_objects_head, draw_points=True, draw_labels = True, scale_factor_distance=200, scale_factor_circle=500)
    frame = circle_frame(frame, 20)
    cv2.imshow("output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.stop()
cv2.destroyAllWindows()



