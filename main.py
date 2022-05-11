"""Main entrypoint for this application"""
import warnings
import logging

import yolov5
from vidgear.gears import CamGear
from vidgear.gears import WriteGear

from environs import Env

env = Env()

SOURCE_STREAM: str = env("SOURCE_STREAM")
SINK_STREAM: str = env("SINK_STREAM")
YOLOV5_MODEL: str = env("YOLOV5_MODEL", default="yolov5s.pt")
YOLOV5_CONFIDENCE_THRESHOLD: float = env.float(
    "YOLOV5_CONFIDENCE_THRESHOLD", default=0.5, validate=lambda x: 0.0 <= x <= 1.0
)
YOLOV5_MAX_DETECTIONS: int = env.int(
    "YOLOV5_MAX_DETECTIONS", default=10, validate=lambda x: x > 0
)
LOG_LEVEL = env.log_level("LOG_LEVEL", logging.WARNING)

verbose = LOG_LEVEL <= logging.INFO

# Setup logger
logging.basicConfig(level=LOG_LEVEL)
logging.captureWarnings(True)
warnings.filterwarnings("once")
LOGGER = logging.getLogger("crowsnest-processor-video-yolov5")

### Configure yolov5 model
MODEL = yolov5.load(YOLOV5_MODEL)

MODEL.conf = 0.25  # NMS confidence threshold
MODEL.iou = 0.45  # NMS IoU threshold
MODEL.agnostic = False  # NMS class-agnostic
MODEL.multi_label = False  # NMS multiple labels per box
MODEL.max_det = 10  # maximum number of detections per image


if __name__ == "__main__":

    try:

        # Open source stream, we always want the last frame since we
        # cant guarantee to keep up with the source frame rate
        source = CamGear(
            source=SOURCE_STREAM,
            colorspace="COLOR_BGR2RGB",
            logging=verbose,
            **{"THREADED_QUEUE_MODE": False}
        ).start()

        sink = WriteGear(
            output_filename=SINK_STREAM,
            logging=verbose,
            **{"-f": "rtsp", "-rtsp_transport": "tcp"}
        )

        # Frame-wise loop
        while True:
            frame = source.read()
            if frame is None:
                break

            # Do object detection
            # pylint: disable=not-callable
            result: yolov5.models.common.Detections = MODEL(
                frame, size=max(frame.shape)
            )

            annotated_frame = result.render()[0]
            sink.write(annotated_frame)

    except Exception:  # pylint: disable=broad-except
        LOGGER.exception("Fundamental failure...")

    finally:
        source.stop()
        sink.close()
