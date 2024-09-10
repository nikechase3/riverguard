import asyncio
import json
import cv2
import numpy as np
import tensorflow as tf
import logging
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, MediaStreamTrack
from av import VideoFrame
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TFLiteInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.height = self.input_shape[1]
        self.width = self.input_shape[2]
        logger.info(f"Model loaded: {self.input_details} and {self.output_details}")

    def preprocess_image(self, frame):
        resized = cv2.resize(frame, (self.width, self.height))
        if resized.dtype != np.uint8:
            resized = (resized * 255).astype(np.uint8)
        input_data = np.expand_dims(resized, axis=0)
        return input_data

    def run_inference(self, input_data):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[1]['index'])
        classes = self.interpreter.get_tensor(self.output_details[0]['index'])
        scores = self.interpreter.get_tensor(self.output_details[3]['index'])
        num_detections = self.interpreter.get_tensor(self.output_details[2]['index'])

        return boxes, classes, scores, num_detections

    def process_single_frame(self, frame):
        original_height, original_width = frame.shape[:2]
        input_data = self.preprocess_image(frame)
        boxes, classes, scores, num_detections = self.run_inference(input_data)

        num_detections = int(num_detections[0])
        detections = []
        for j in range(num_detections):
            score = scores[0, j]
            if score >= 7:  # Adjust threshold if necessary
                ymin, xmin, ymax, xmax = boxes[0, j]
                xmin = int(xmin * original_width)
                xmax = int(xmax * original_width)
                ymin = int(ymin * original_height)
                ymax = int(ymax * original_height)

                # Calculate bounding box center
                bbox_center_x = (xmin + xmax) // 2
                frame_center_x = original_width // 2

                # Determine direction based on bounding box position
                if abs(bbox_center_x - frame_center_x) < 50:  # Tolerance for "center" make bigger if you have bigger center
                    direction = "forward"
                elif bbox_center_x < frame_center_x:
                    direction = "left"
                else:
                    direction = "right"

                detections.append({
                    "class": int(classes[0, j]),
                    "score": float(score),
                    "bbox": [xmin, ymin, xmax, ymax],
                    "direction": direction  # Add the direction to the detection
                })
                logger.info(f"Detection appended: {detections[-1]}")
        return detections


class TFLiteInferencePool:
    def __init__(self, model_path, pool_size=4):
        self.pool = Queue(maxsize=pool_size)
        for _ in range(pool_size):
            interpreter = TFLiteInference(model_path)
            self.pool.put(interpreter)

    def acquire(self):
        return self.pool.get()

    def release(self, interpreter):
        self.pool.put(interpreter)


tflite_pool = TFLiteInferencePool('/PATH/TO/MODEL')

# ThreadPoolExecutor for running inference
executor = ThreadPoolExecutor(max_workers=10)


class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track, inference_pool, data_channel):
        super().__init__()
        self.track = track
        self.inference_pool = inference_pool
        self.data_channel = data_channel
        self.frame_buffer = []

    async def run_inference_async(self, frame_buffer):
        loop = asyncio.get_event_loop()
        interpreter = self.inference_pool.acquire()
        try:
            # Perform inference on the frame buffer in a separate thread
            detections_batch = await loop.run_in_executor(
                executor,
                lambda: [interpreter.process_single_frame(frame) for frame in frame_buffer]
            )
        finally:
            self.inference_pool.release(interpreter)
        return detections_batch

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="rgb24")
        self.frame_buffer.append(img)

        if len(self.frame_buffer) >= 4:  # Adjust batch size if necessary
            detections_batch = await self.run_inference_async(self.frame_buffer)
            self.frame_buffer.clear()

            for detections in detections_batch:
                for detection in detections:
                    bbox = detection.get("bbox")
                    detection["bbox"] = [int(coord) for coord in bbox]
                    detection["score"] = float(detection["score"])
                    detection["class"] = int(detection["class"])
                    # Send the direction (left, right, forward) with detections
                try:
                    if self.data_channel.readyState == "open":
                        self.data_channel.send(json.dumps(detections))
                except (TypeError, ValueError, OverflowError) as e:
                    logger.error(f"Error serializing detections: {e}")
                    logger.error(f"Failed to serialize detections: {detections}")

        return frame


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection(configuration=RTCConfiguration(
        iceServers=[RTCIceServer(urls=['stun:stun.l.google.com:19302'])]
    ))
    pcs.add(pc)

    data_channel = pc.createDataChannel("detections", ordered=True, negotiated=True, id=0)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            local_video = VideoTransformTrack(track, tflite_pool, data_channel)
            pc.addTrack(local_video)

    @data_channel.on("open")
    async def on_open():
        logger.info("Data channel opened")

    @data_channel.on("close")
    async def on_close():
        logger.info("Data channel closed")

    @data_channel.on("message")
    async def on_message(message):
        logger.info(f"Received message: {message}")
        # Handle received messages here

    @data_channel.on("error")
    async def on_error(error):
        logger.error(f"Data channel error: {error}")

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


pcs = set()

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    web.run_app(app, host='0.0.0.0', port=YOUROPENPORT)
