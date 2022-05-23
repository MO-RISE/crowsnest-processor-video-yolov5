"""Main entrypoint for this application"""
import warnings
import logging
import time
from datetime import datetime, timezone

import yolov5
from vidgear.gears import CamGear, WriteGear

from environs import Env

env = Env()

SOURCE_STREAM: str = env("SOURCE_STREAM")
SINK_STREAM: str = env("SINK_STREAM")
YOLOV5_MODEL: str = env("YOLOV5_MODEL", default="yolov5s.pt")
YOLOV5_CONFIDENCE_THRESHOLD: float = env.float(
    "YOLOV5_CONFIDENCE_THRESHOLD", default=0.5, validate=lambda x: 0.0 <= x <= 1.0
)
YOLOV5_MAX_DETECTIONS: int = env.int(
    "YOLOV5_MAX_DETECTIONS", default=100, validate=lambda x: x > 0
)
YOLOV5_DEVICE: str = env("YOLOV5_DEVICE", default="cpu")
YOLOV5_FRAMERATE: float = env.float("YOLOV5_FRAMERATE", default=1)

LOG_LEVEL = env.log_level("LOG_LEVEL", logging.WARNING)
FFMPEG_OUTPUT = env.bool("FFMPEG_OUTPUT", default=False)


verbose = LOG_LEVEL <= logging.INFO

# Setup logger
logging.basicConfig(level=LOG_LEVEL, force=True)
logging.captureWarnings(True)
warnings.filterwarnings("ignore")
LOGGER = logging.getLogger("crowsnest-processor-video-yolov5")

### Configure yolov5 model
MODEL = yolov5.load(YOLOV5_MODEL, device=YOLOV5_DEVICE, verbose=verbose)

MODEL.conf = YOLOV5_CONFIDENCE_THRESHOLD  # NMS confidence threshold
MODEL.iou = 0.45  # NMS IoU threshold
MODEL.agnostic = False  # NMS class-agnostic
MODEL.multi_label = False  # NMS multiple labels per box
MODEL.max_det = YOLOV5_MAX_DETECTIONS  # maximum number of detections per image


try:

    # Open source stream, we always want the last frame since we
    # cant guarantee to keep up with the source frame rate
    source = CamGear(
        source=SOURCE_STREAM,
        colorspace="COLOR_BGR2RGB",
        logging=verbose or FFMPEG_OUTPUT,
        **{"THREADED_QUEUE_MODE": False, "-fflags": "nobuffer"}
    ).start()

    sink = WriteGear(
        output_filename=SINK_STREAM,
        logging=verbose or FFMPEG_OUTPUT,
        **{
            "-f": "rtsp",
            "-rtsp_transport": "tcp",
            "-tune": "zerolatency",
            "-preset": "ultrafast",
            "-stimeout": "1000000",
            "-input_framerate": YOLOV5_FRAMERATE,
            "-r": source.framerate,
        }
    )

    # Frame-wise loop
    while True:

        frame_t0 = time.time()

        frame = source.read()
        if frame is None:
            break

        LOGGER.debug("New frame read at: %s", datetime.now(timezone.utc))

        t0 = time.time()

        # Do object detection
        # pylint: disable=not-callable
        result: yolov5.models.common.Detections = MODEL(frame, size=max(frame.shape))

        LOGGER.debug("Inference took %s seconds", time.time() - t0)
        LOGGER.debug(result.print())

        annotated_frame = result.render()[0]

        LOGGER.debug("Annotated frame rendered with shape: %s", annotated_frame.shape)
        sink.write(frame, rgb_mode=True)

        time_to_sleep = 1 / YOLOV5_FRAMERATE - (time.time() - frame_t0)

        if time_to_sleep < 0:
            LOGGER.warning("YOLOV5_FRAMERATE not reached!")
        else:
            time.sleep(time_to_sleep)

except Exception:  # pylint: disable=broad-except
    LOGGER.exception("Fundamental failure...")

finally:
    source.stop()
    sink.close()
