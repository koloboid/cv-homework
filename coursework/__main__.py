import asyncio
import logging
import os

from injector import Injector

from coursework.config import Config
from coursework.frame import Frame
from coursework.opencv_capture import OpenCVCapture
from coursework.opencv_output import OpenCVDisplayOutput
from coursework.processing_draw import ProcessingDraw
from coursework.processing_lane import ProcessingLane
from coursework.processing_sign import ProcessingSign


logging.basicConfig(
    level=logging.DEBUG if bool(os.environ.get("DEBUG")) else logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
)
logger = logging.getLogger("MAIN")


async def main() -> None:
    logger.info("Starting coursework application")
    capture = None
    output = None
    config = Config()
    try:
        injector = Injector()
        injector.binder.bind(Config, to=config)
        capture = injector.get(OpenCVCapture)
        processing_lanes = injector.get(ProcessingLane)
        processing_signs = injector.get(ProcessingSign)
        processing_draw = injector.get(ProcessingDraw)
        output = injector.get(OpenCVDisplayOutput)

        def frame_handler(frame: Frame) -> None:
            processing_lanes.handle_frame(frame)
            processing_signs.handle_frame(frame)
            processing_draw.handle_frame(frame)
            output.handle_frame(frame)

        logger.debug("Initialization")
        await capture.init(frame_handler)
        await output.init()
        await capture.run()
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down")
        if capture is not None:
            await capture.stop()
        if output is not None:
            await output.stop()


if __name__ == "__main__":
    asyncio.run(main())
