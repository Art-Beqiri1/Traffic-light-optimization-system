"""Shared config migration/validation for the web and desktop apps."""
import copy
import json
import os
from live_controller import DEFAULT_CONTROL, validate_control

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, 'live_config.json')
DIRECTIONS = ('N', 'S', 'E', 'W')
DEFAULT_CONFIG = dict(rtsp_url='', confidence=.35, line_position=.55,
    grid_rows=2, grid_cols=2, dqn_episodes=120, model_name='yolov8n.pt',
    imgsz=640, rtsp_transport='tcp', enhance=True, control=DEFAULT_CONTROL)


def normalize_cameras(items):
    if not isinstance(items, list) or len(items) > 4:
        raise ValueError('Configure up to four cameras')
    result = []
    seen_ids, seen_approaches = set(), set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError('Each camera must be an object')
        cid = item.get('id', DIRECTIONS[i])
        approach = item.get('approach', cid)
        url = item.get('rtsp_url', '')
        if cid not in DIRECTIONS or cid in seen_ids:
            raise ValueError('Camera IDs must be unique N, S, E or W slots')
        if approach not in (*DIRECTIONS, 'ALL'):
            raise ValueError('Choose N, S, E, W or ALL for the camera approach')
        if not isinstance(url, str):
            raise ValueError('Camera source must be text')
        enabled = item.get('enabled', True)
        if not isinstance(enabled, bool):
            raise ValueError('Camera enabled must be true or false')
        url = url.strip()
        if enabled and url:
            if approach in seen_approaches or ('ALL' in seen_approaches) or (approach == 'ALL' and seen_approaches):
                raise ValueError('Use one camera per approach; Whole intersection must be used alone')
            seen_approaches.add(approach)
        seen_ids.add(cid)
        result.append(dict(id=cid, approach=approach, rtsp_url=url, enabled=enabled))
    for cid in DIRECTIONS:
        if cid not in seen_ids:
            result.append(dict(id=cid, approach=cid, rtsp_url='', enabled=True))
    return sorted(result, key=lambda c: DIRECTIONS.index(c['id']))


def load_config():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    try:
        with open(CONFIG_FILE, encoding='utf-8') as f:
            cfg.update(json.load(f))
    except (OSError, ValueError):
        pass
    # Old single-camera configurations retain their original quadrant zones.
    cfg['cameras'] = normalize_cameras(cfg.get('cameras', [dict(id='N', approach='ALL', rtsp_url=cfg['rtsp_url'])] if cfg['rtsp_url'] else []))
    cfg['control'] = validate_control(cfg.get('control', {}))
    return cfg


def save_config(cfg):
    # Atomic replace prevents partially written JSON after interruption.
    import tempfile
    fd, path = tempfile.mkstemp(dir=BASE_DIR, suffix='.json.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)
        os.replace(path, CONFIG_FILE)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def update_settings(config, data):
    if not isinstance(data, dict):
        raise ValueError('Settings must be an object')
    cfg = copy.deepcopy(config)
    if 'rtsp_url' in data and 'cameras' not in data:
        cfg['cameras'][0]['rtsp_url'] = data['rtsp_url']
    if 'cameras' in data:
        cfg['cameras'] = data['cameras']
    cfg['cameras'] = normalize_cameras(cfg['cameras'])
    cfg['rtsp_url'] = cfg['cameras'][0]['rtsp_url']
    models = {'fast': 'yolov8n.pt', 'balanced': 'yolov8s.pt', 'accurate': 'yolov8m.pt'}
    if 'model_quality' in data:
        if data['model_quality'] not in models:
            raise ValueError('Unknown detection model')
        cfg['model_name'] = models[data['model_quality']]
    for key in ('confidence', 'line_position'):
        if key in data:
            if isinstance(data[key], bool) or not isinstance(data[key], (int, float)) or not 0 < data[key] < 1:
                raise ValueError(f'{key} must be between 0 and 1')
            cfg[key] = data[key]
    if 'imgsz' in data:
        if data['imgsz'] not in (480, 640, 960, 1280):
            raise ValueError('Unsupported detection resolution')
        cfg['imgsz'] = data['imgsz']
    if 'rtsp_transport' in data:
        if data['rtsp_transport'] not in ('tcp', 'udp'):
            raise ValueError('Transport must be tcp or udp')
        cfg['rtsp_transport'] = data['rtsp_transport']
    if 'enhance' in data:
        if not isinstance(data['enhance'], bool):
            raise ValueError('Enhance must be true or false')
        cfg['enhance'] = data['enhance']
    return cfg
