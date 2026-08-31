"""Run: python -m unittest test_live_features -v. No physical cameras or model downloads.
Capture/inference are substituted; controller, vision wrapper and HTTP are real.
"""
import copy
import random
import threading
import time
import unittest
from unittest.mock import patch
from signal_engine import SignalSequencer
from live_controller import LiveTrafficController
from live_settings import DEFAULT_CONFIG, normalize_cameras
from camera_manager import CameraManager
from vision import VehicleVision, VisionState

class Signals(unittest.TestCase):
    def test_full_sequence_and_min_green(self):
        s = SignalSequencer(0)
        for t,target,stage in [(1.9,0,'ALL_RED'),(2,0,'GREEN'),(8.9,1,'GREEN'),(9,1,'YELLOW'),(11.9,1,'YELLOW'),(12,1,'ALL_RED'),(13.9,1,'ALL_RED'),(14,1,'GREEN')]:
            s.advance(t,target); self.assertEqual(s.stage,stage)
        self.assertEqual(s.snapshot(14)['ew'],'GREEN')
    def test_delayed_tick_never_skips_clearance(self):
        s=SignalSequencer(0);s.advance(2,0);s.advance(9,1);s.advance(1000,1)
        self.assertEqual(s.stage,'ALL_RED')
        s.advance(1001,1);self.assertEqual(s.stage,'ALL_RED')
        s.advance(1002,1);self.assertEqual(s.stage,'GREEN')
    def test_timing_edit_does_not_shorten_active_yellow(self):
        s=SignalSequencer(0,yellow=8);s.advance(2,0);s.advance(9,1);s.yellow=1
        s.advance(11,1);self.assertEqual(s.stage,'YELLOW')
        s.advance(17,1);self.assertEqual(s.stage,'ALL_RED')
    def test_rapid_requests_preserve_clearance(self):
        s=SignalSequencer(0);s.advance(2,0);s.advance(9,1)
        for t,d in [(9.1,0),(9.2,None),(10,1),(11,0)]:
            s.advance(t,d);self.assertEqual(s.stage,'YELLOW')
        s.advance(12,None);s.advance(14,None);self.assertEqual(s.stage,'ALL_RED')

class Control(unittest.TestCase):
    def make(self,**kwargs):return LiveTrafficController(rows=1,cols=1,clock=lambda:0,**kwargs)
    def test_fixed_unequal_durations(self):
        c=self.make(control_config={'mode':'fixed','fixed_ns_seconds':7,'fixed_ew_seconds':9})
        for t,p in [(0,'ALL_RED'),(2,'NS_GREEN'),(9,'NS_YELLOW'),(12,'ALL_RED'),(14,'EW_GREEN'),(22,'EW_GREEN'),(23,'EW_YELLOW'),(26,'ALL_RED'),(28,'NS_GREEN')]:
            c._tick(t);self.assertEqual(c.get_camera_signal()['phase'],p)
    def test_manual_startup_red_and_indefinite_hold(self):
        c=self.make(control_config={'mode':'manual'});c._tick(200)
        self.assertEqual(c.get_camera_signal()['phase'],'ALL_RED')
        c.configure({'manual_phase':'NS_GREEN'});c._tick(201);c._tick(1000)
        self.assertEqual(c.get_camera_signal()['phase'],'NS_GREEN')
        c.configure({'manual_phase':'EW_GREEN'});c._tick(1001)
        self.assertEqual(c.get_camera_signal()['phase'],'NS_YELLOW')
        c._tick(1004);c._tick(1006);self.assertEqual(c.get_camera_signal()['phase'],'EW_GREEN')
    def test_manual_all_red_uses_yellow(self):
        c=self.make(control_config={'mode':'fixed'});c._tick(2)
        c.configure({'mode':'manual','manual_phase':'ALL_RED'});c._tick(3)
        self.assertEqual(c.get_camera_signal()['phase'],'NS_YELLOW')
        c._tick(6);c._tick(100);self.assertEqual(c.get_camera_signal()['phase'],'ALL_RED')
    def test_mode_change_during_yellow(self):
        c=self.make(control_config={'mode':'fixed','fixed_ns_seconds':7});c._tick(2);c._tick(9)
        c.configure({'mode':'manual','manual_phase':'EW_GREEN'});c._tick(10)
        self.assertEqual(c.get_camera_signal()['phase'],'NS_YELLOW')
        c._tick(12);c._tick(14);self.assertEqual(c.get_camera_signal()['phase'],'EW_GREEN')
    def test_camera_fault_latches_fallback(self):
        h={'configured':4,'healthy':False,'reason':'N stale'}
        c=self.make(get_camera_health=lambda:h);c._tick(0)
        self.assertEqual(c.effective_mode,'fixed')
        h['healthy']=True;c._tick(2);self.assertEqual(c.effective_mode,'fixed')
        c.configure({'mode':'ai'});self.assertEqual(c.effective_mode,'ai')
    def test_manual_fixed_independent_of_ai_and_camera_health(self):
        for m in ('manual','fixed'):
            c=self.make(control_config={'mode':m},get_camera_health=lambda:{'configured':1,'healthy':False,'reason':'offline'})
            with patch.object(c,'_submit_ai',side_effect=AssertionError('AI called')):c._tick(2);c._tick(80)
            self.assertEqual(c.effective_mode,m);self.assertFalse(c.fallback_reason)
    def test_ai_errors_invalid_outputs(self):
        for answer in (None,{(0,0):float('nan')},{},{(0,0):2}):
            class Policy:
                name='bad'
                def act(self,net):
                    if answer is None:raise RuntimeError('oops')
                    return answer
            c=self.make();c.trained_policy=Policy();c._tick(0);c._ai_thread.join(1);c._tick(.1)
            self.assertEqual(c.effective_mode,'fixed');self.assertIn('failed',c.fallback_reason)
    def test_hung_ai_and_late_result_do_not_block_manual(self):
        release=threading.Event()
        class Policy:
            name='stalled'
            def act(self,net):release.wait(5);return {(0,0):1}
        c=self.make();c.trained_policy=Policy()
        try:
            c._tick(0);thread=c._ai_thread;c._tick(3.1)
            self.assertEqual(c.effective_mode,'fixed')
            c.configure({'mode':'manual','manual_phase':'NS_GREEN'});c._tick(4)
            self.assertEqual(c.get_camera_signal()['phase'],'NS_GREEN')
            c.configure({'mode':'ai'});c._tick(5);self.assertIs(c._ai_thread,thread)
            c.configure({'mode':'manual','manual_phase':'ALL_RED'})
        finally:release.set();c._ai_thread.join(1)
        c._tick(6);self.assertEqual(c.effective_mode,'manual')
        self.assertEqual(c.get_camera_signal()['target'],'ALL_RED')
    def test_yellow_all_red_no_discharge_live_queues_unpolluted(self):
        live=dict(N=10,S=7,E=4,W=2)
        c=self.make(control_config={'mode':'fixed','fixed_ns_seconds':7},get_live_queues=lambda:live)
        c._tick(2);c._tick(9);served=c.net.served_total()
        for t in [10,11,12,13]:c._tick(t)
        self.assertEqual(c.net.served_total(),served)
        self.assertEqual(c.snapshot()['intersections'][0]['queues'],live)
    def test_no_conflicting_greens_random_commands(self):
        r=random.Random(91);c=self.make(control_config={'mode':'manual'})
        for i in range(3000):
            if i%3==0:c.configure({'manual_phase':r.choice(['NS_GREEN','EW_GREEN','ALL_RED'])})
            c._tick(i*.2);s=c.get_camera_signal()
            self.assertFalse(s['ns'] in ('GREEN','YELLOW') and s['ew'] in ('GREEN','YELLOW'))
    def test_invalid_settings_atomic(self):
        c=self.make();before=copy.deepcopy(c.control)
        for d in ({'mode':'other'},{'yellow_seconds':0},{'fixed_ns_seconds':6},{'all_red_seconds':float('nan')},{'fixed_ew_seconds':True},{'manual_phase':'EW_GREEN'}):
            with self.assertRaises(ValueError):c.configure(d)
            self.assertEqual(c.control,before)

class FakeVision:
    def __init__(self,rtsp_url='',**kwargs):self.state=VisionState(rtsp_url=rtsp_url);self._running=False;self.kwargs=kwargs
    def start(self):self._running=True
    def stop(self):self._running=False

class Cameras(unittest.TestCase):
    def config(self):
        cfg=copy.deepcopy(DEFAULT_CONFIG)
        cfg['cameras']=normalize_cameras([dict(id=d,approach=d,rtsp_url='test-'+d) for d in ('N','S','E','W')])
        return cfg
    def healthy(self,v,queues):
        v.state.camera_status='online';v.state.detection_ok=True;v.state.last_detection_at=time.monotonic();v.state.queues=queues
    def test_four_independent_trackers_correct_mapping(self):
        m=CameraManager(self.config(),factory=FakeVision);m.start()
        for i,d in enumerate(('N','S','E','W')):self.healthy(m.get(d),{a:(i+1 if a==d else 999) for a in ('N','S','E','W')})
        self.assertEqual(len({id(v) for v in m.visions.values()}),4)
        self.assertEqual(m.live_queues(),dict(N=1,S=2,E=3,W=4));self.assertTrue(m.health()['healthy'])
    def test_stale_detections_excluded(self):
        m=CameraManager(self.config(),factory=FakeVision);m.start()
        for d in ('N','S','E','W'):self.healthy(m.get(d),dict(N=1,S=1,E=1,W=1))
        m.started={};m.get('E').state.last_detection_at=time.monotonic()-6
        self.assertEqual(m.live_queues()['E'],0);self.assertFalse(m.health()['healthy']);self.assertIn('E',m.health()['reason'])
    def test_stop_one_leaves_others_running(self):
        m=CameraManager(self.config(),factory=FakeVision);m.start();m.stop('S')
        self.assertFalse(m.get('S')._running);self.assertTrue(all(m.get(d)._running for d in ('N','E','W')))
    def test_source_change_replaces_and_reconnects(self):
        cfg=self.config();m=CameraManager(cfg,factory=FakeVision);m.start();old=m.get('N');others=[m.get(d) for d in ('S','E','W')]
        updated=copy.deepcopy(cfg);updated['cameras'][0]['rtsp_url']='new-source';m.configure(updated)
        self.assertFalse(old._running);self.assertIsNot(old,m.get('N'));self.assertTrue(m.get('N')._running)
        self.assertEqual(others,[m.get(d) for d in ('S','E','W')])
    def test_validate_fifth_duplicate_and_overlapping_cameras(self):
        self.assertEqual(len(normalize_cameras([])),4)
        for items in ([{'id':'N'}]*5,[{'id':'N'},{'id':'N'}],[{'id':'N','rtsp_url':'x','approach':'ALL'},{'id':'S','rtsp_url':'y'}],[{'id':'N','rtsp_url':'x'},{'id':'S','approach':'N','rtsp_url':'y'}]):
            with self.assertRaises(ValueError):normalize_cameras(items)
    def test_legacy_whole_intersection(self):
        cfg=self.config();cfg['cameras']=normalize_cameras([dict(id='N',approach='ALL',rtsp_url='legacy')]);m=CameraManager(cfg,factory=FakeVision);m.start()
        self.healthy(m.get('N'),dict(N=3,S=6,E=9,W=12));self.assertEqual(m.live_queues(),dict(N=3,S=6,E=9,W=12))
    def test_detection_failure_vs_successful_empty_frame(self):
        import numpy as np
        class Model:
            def __init__(self,broken):self.broken=broken
            def track(self,*a,**kw):
                if self.broken:raise RuntimeError('broken')
                return []
        for broken in (True,False):
            v=VehicleVision(enhance=False);v._model=Model(broken);v._process_frame(np.zeros((120,160,3),dtype=np.uint8))
            self.assertEqual(v.state.snapshot()['detection_ok'],not broken)
            self.assertEqual(v.state.snapshot()['last_detection_age_s'] is None,broken)
    def test_zero_area_zone_never_counts(self):self.assertFalse(VehicleVision()._in_zone(0,0,(0,0,0,0)))

class HTTP(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp.test_utils import TestClient,TestServer
        import live_app
        self.module=live_app;self.saved_config=copy.deepcopy(live_app.config)
        self.patches=[patch.object(live_app,'controller',LiveTrafficController(control_config={'mode':'manual'})),patch.object(live_app,'save_config')]
        for p in self.patches:p.start()
        self.client=TestClient(TestServer(live_app.build_app(start_workers=False)));await self.client.start_server()
    async def asyncTearDown(self):
        await self.client.close();self.module.config=self.saved_config
        for p in reversed(self.patches):p.stop()
    async def test_state_and_manual_command(self):
        state=await(await self.client.get('/api/state')).json();self.assertEqual(len(state['cameras']),4)
        r=await self.client.post('/api/control',json={'mode':'manual','manual_phase':'EW_GREEN'})
        self.assertEqual(r.status,200);self.assertEqual((await r.json())['control']['manual_phase'],'EW_GREEN')
    async def test_invalid_control_returns_400(self):
        for data in ({'mode':'bad'},{'yellow_seconds':0},{'mode':'fixed','manual_phase':'NS_GREEN'}):
            r=await self.client.post('/api/control',json=data);self.assertEqual(r.status,400)
        self.assertEqual(self.module.controller.effective_mode,'manual')
    async def test_unknown_camera_and_cross_origin_rejected(self):
        r=await self.client.post('/api/camera/start?camera_id=BAD');self.assertEqual(r.status,400)
        r=await self.client.post('/api/control',json={'mode':'fixed'},headers={'Origin':'https://outside.invalid'});self.assertEqual(r.status,403)
    async def test_four_mjpeg_feeds_do_not_block_commands(self):
        responses=[]
        try:
            for cid in ('N','S','E','W'):
                r=await self.client.get('/video_feed?camera_id='+cid);responses.append(r);self.assertIn(b'--frame',await r.content.read(64))
            start=time.monotonic();r=await self.client.post('/api/control',json={'mode':'fixed'})
            self.assertEqual(r.status,200);self.assertLess(time.monotonic()-start,1.5)
        finally:
            for r in responses:r.close()
    async def test_fifth_camera_rejected(self):
        r=await self.client.post('/api/settings',json={'cameras':[{'id':'N'}]*5});self.assertEqual(r.status,400)

if __name__=='__main__':unittest.main()
