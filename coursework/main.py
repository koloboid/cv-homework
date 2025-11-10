import asyncio
from injector import Injector

from coursework.frame import Frame
from coursework.opencv_capture_input import OpenCVCaptureInput


async def main():

    try:
        injector = Injector()
        input = injector.get(OpenCVCaptureInput)
        processing_lanes = injector.get(ProcessingLanes)
        processing_signs = injector.get(ProcessingSigns)
        processing_draw = injector.get(ProcessingDraw)
        output = injector.get(OpenCVDisplayOutput)

        async def frame_handler(frame: Frame):
            await processing_lanes.process(frame)
            await processing_signs.process(frame)
            await processing_draw.process(frame)
            await output.send(frame)

        await input.init(frame_handler)
        await output.init()
        await input.start()
        await asyncio.sleep(0)
    except asyncio.exceptions.CancelledError:
        await input.stop()
        await output.stop()


if __name__ == "__main__":
    asyncio.run(main())
