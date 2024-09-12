import asyncio
import json
import cv2
import av
import time
import serial  # Import pyserial for serial communication
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration, RTCIceServer
import aiohttp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Establish serial communication with Arduino
ser = serial.Serial('/dev/ttyUSB0', 9600)  # Replace '/dev/ttyUSB0' with your actual serial port

class CameraVideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.last_frame_time = 0
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps

    async def recv(self):
        while True:
            current_time = time.time()
            if current_time - self.last_frame_time >= self.frame_interval:
                pts, time_base = await self.next_timestamp()
                ret, frame = self.cap.read()
                if not ret:
                    raise RuntimeError("Could not read from camera")
                frame = cv2.resize(frame, (512, 1024))
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                compressed_frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                rgb_frame = cv2.cvtColor(compressed_frame, cv2.COLOR_BGR2RGB)
                video_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
                video_frame.pts = pts
                video_frame.time_base = time_base
                self.last_frame_time = current_time
                return video_frame
            await asyncio.sleep(0.001)

async def run_offer(pc, session):
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    async with session.post('http://YOURIP:PORT/offer', json={
        'sdp': pc.localDescription.sdp,
        'type': pc.localDescription.type
    }) as response:
        answer = await response.json()
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer['sdp'], type=answer['type']))

async def wait_for_ice_connection(pc):
    logger.info("Waiting for ICE connection...")
    for _ in range(30):
        if pc.iceConnectionState in ["completed", "connected"]:
            logger.info("ICE connection established")
            return True
        await asyncio.sleep(1)
    logger.warning("ICE connection timed out")
    return False

async def maintain_connection(pc, session):
    while True:
        if pc.connectionState != "connected":
            logger.info("Connection lost. Attempting to reconnect...")
            await run_offer(pc, session)
            if not await wait_for_ice_connection(pc):
                logger.error("Failed to re-establish connection")
                return
        await asyncio.sleep(5)

async def main():
    pc = RTCPeerConnection(configuration=RTCConfiguration(
        iceServers=[RTCIceServer(urls=['stun:stun.l.google.com:19302'])]
    ))

    video_track = CameraVideoStreamTrack()
    pc.addTrack(video_track)

    channel = pc.createDataChannel("detections", ordered=True, negotiated=True, id=0)

    @channel.on("open")
    def on_open():
        logger.info("Data channel opened")

    @channel.on("close")
    def on_close():
        logger.info("Data channel closed")

    @channel.on("message")
    def on_message(message):
        detections = json.loads(message)
        for detection in detections:
            direction = detection.get("direction")
            if direction:
                logger.info(f"Move {direction}")
                # Send the command to Arduino
                ser.write(f"{direction}\n".encode())  # Send the direction command to Arduino via serial
            else:
                logger.info(f"Detection received: {detection}")

    @channel.on("error")
    def on_error(error):
        logger.error(f"Data channel error: {error}")

    @channel.on("statechange")
    def on_state_change():
        logger.info(f"Data channel state changed to: {channel.readyState}")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            logger.info(f'Connection state is {pc.connectionState}')

    @pc.on("signalingstatechange")
    async def on_signalingstatechange():
        logger.info(f"Signaling state is {pc.signalingState}")

    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange():
        logger.info(f"ICE gathering state is {pc.iceGatheringState}")

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state is {pc.iceConnectionState}")

    async with aiohttp.ClientSession() as session:
        try:
            await run_offer(pc, session)
            logger.info("Offer sent, waiting for ICE connection...")

            if not await wait_for_ice_connection(pc):
                logger.error("Failed to establish initial connection")
                return

            logger.info("Connection established")

            # Start the connection maintenance task
            maintenance_task = asyncio.create_task(maintain_connection(pc, session))

            # Create a main loop task
            main_loop = asyncio.create_task(asyncio.sleep(float('inf')))

            # Wait for the main loop to be cancelled
            try:
                await main_loop
            except asyncio.CancelledError:
                logger.info("Main loop cancelled, closing connection")

        except Exception as e:
            logger.error(f"Error in main: {str(e)}")
        finally:
            if 'maintenance_task' in locals():
                maintenance_task.cancel()
            # Ensure the peer connection is closed properly
            await pc.close()
            logger.info("Peer connection closed")

if __name__ == '__main__':
    asyncio.run(main())