"""FlowMind Live browser dashboard. Run: python live_app.py"""
import asyncio
import json
import os

import cv2
import numpy as np
from aiohttp import web
from camera_manager import CameraManager
from live_settings import BASE_DIR, load_config, save_config, update_settings
from live_controller import LiveTrafficController

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from webrtc_stream import LiveDetectionTrack
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

config = load_config()
cameras = CameraManager(config)
controller = LiveTrafficController(rows=config['grid_rows'], cols=config['grid_cols'],
    dqn_episodes=config['dqn_episodes'], get_live_queues=cameras.live_queues,
    get_camera_health=cameras.health, control_config=config['control'])
cameras.set_signal_callback(controller.get_camera_signal)
active_peers = set()
settings_lock = asyncio.Lock()


@web.middleware
async def errors(request, handler):
    try:
        if request.method == 'POST':
            origin = request.headers.get('Origin')
            if origin and origin != f'{request.scheme}://{request.host}':
                return web.json_response({'error': 'Cross-origin commands are disabled'}, status=403)
        return await handler(request)
    except (ValueError, TypeError, KeyError) as exc:
        return web.json_response({'error': str(exc)}, status=400)
    except RuntimeError as exc:
        return web.json_response({'error': str(exc)}, status=409)


async def index(request):
    with open(os.path.join(BASE_DIR, 'templates', 'dashboard.html'), encoding='utf-8') as f:
        html = f.read()
    return web.Response(text=html.replace('{{ webrtc_available }}', 'true' if WEBRTC_AVAILABLE else 'false'),
                        content_type='text/html')


async def offer(request):
    if not WEBRTC_AVAILABLE:
        return web.json_response({'error': 'WebRTC unavailable; use MJPEG'}, status=503)
    params = await request.json()
    cid = params.get('camera_id', 'N')
    cameras.get(cid)
    pc = RTCPeerConnection()
    active_peers.add(pc)
    @pc.on('connectionstatechange')
    async def changed():
        if pc.connectionState in ('failed', 'closed', 'disconnected'):
            active_peers.discard(pc)
            await pc.close()
    try:
        pc.addTrack(LiveDetectionTrack(lambda: cameras.get(cid), fps=20))
        await pc.setRemoteDescription(RTCSessionDescription(sdp=params['sdp'], type=params['type']))
        await pc.setLocalDescription(await pc.createAnswer())
        return web.json_response({'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type})
    except Exception:
        active_peers.discard(pc)
        await pc.close()
        raise


def jpeg_frame(cid):
    v = cameras.get(cid)
    frame = v.get_latest_frame() if v._running else None
    if frame is None:
        frame = np.zeros((240, 420, 3), dtype=np.uint8)
        cv2.putText(frame, f'{cid}: waiting for camera', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, .6, (180, 180, 180), 1)
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n' if ok else b''


async def video_feed(request):
    cid = request.query.get('camera_id', 'N')
    cameras.get(cid)
    response = web.StreamResponse(headers={'Content-Type': 'multipart/x-mixed-replace; boundary=frame', 'Cache-Control': 'no-store'})
    await response.prepare(request)
    try:
        while True:
            # Capture/encode waits never block API commands, even with four feeds.
            await response.write(await asyncio.to_thread(jpeg_frame, cid))
            await asyncio.sleep(.1)
    except (ConnectionResetError, ConnectionAbortedError, asyncio.CancelledError):
        pass
    return response


async def api_state(request):
    return web.json_response({'vision': cameras.aggregate(), 'cameras': cameras.snapshot(),
                              'traffic': controller.snapshot(), 'settings': config},
                             headers={'Cache-Control': 'no-store'})


async def api_settings(request):
    global config
    data = await request.json()
    async with settings_lock:
        updated = update_settings(config, data)
        await asyncio.to_thread(cameras.configure, updated)
        # Control commands can complete while cameras reconnect.
        updated['control'] = dict(controller.control)
        save_config(updated)
        config = updated
    return web.json_response({'ok': True, 'config': config})


async def api_control(request):
    result = controller.configure(await request.json())
    config['control'] = dict(controller.control)
    save_config(config)
    return web.json_response({'ok': True, 'control': result})


async def api_camera_start(request):
    await asyncio.to_thread(cameras.start, request.query.get('camera_id'))
    return web.json_response({'ok': True})


async def api_camera_stop(request):
    await asyncio.to_thread(cameras.stop, request.query.get('camera_id'))
    return web.json_response({'ok': True})


async def on_startup(app):
    controller.start()
    await asyncio.to_thread(cameras.start)


async def on_shutdown(app):
    controller.stop()
    await asyncio.to_thread(cameras.stop)
    await asyncio.gather(*(pc.close() for pc in list(active_peers)))
    active_peers.clear()


def build_app(start_workers=True):
    app = web.Application(middlewares=[errors], client_max_size=64 * 1024)
    app.router.add_get('/', index)
    app.router.add_post('/offer', offer)
    app.router.add_get('/video_feed', video_feed)
    app.router.add_get('/api/state', api_state)
    app.router.add_post('/api/settings', api_settings)
    app.router.add_post('/api/control', api_control)
    app.router.add_post('/api/camera/start', api_camera_start)
    app.router.add_post('/api/camera/stop', api_camera_stop)
    if start_workers:
        app.on_startup.append(on_startup)
        app.on_shutdown.append(on_shutdown)
    return app


app = build_app()
if __name__ == '__main__':
    # No authentication is supplied: local-only by default. Explicit LAN opt-in
    # requires a trusted network/firewall or an authenticated reverse proxy.
    web.run_app(app, host=os.environ.get('FLOWMIND_HOST', '127.0.0.1'), port=5000)
