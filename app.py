#!/usr/bin/env python3
"""Live orchard-row guidance dashboard (simulator or RealSense D455)."""

from __future__ import annotations

import argparse
import threading
import time
from typing import Optional

import cv2
from flask import Flask, Response, jsonify, request

from orchard_guidance.config import GuidanceConfig
from orchard_guidance.pipeline import GuidancePipeline
from orchard_guidance.simulator import OrchardSimulator
from orchard_guidance.visualize import annotate

app = Flask(__name__)
_lock = threading.Lock()
_jpeg: Optional[bytes] = None
_state: dict = {}
_sim: Optional[OrchardSimulator] = None
_pipeline: Optional[GuidancePipeline] = None
_cfg = GuidanceConfig()
_stop = False
_source = "sim"
_source_obj = None

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Orchard row guidance</title>
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' fill='%23141a17'/%3E%3Crect x='7' y='2' width='2' height='12' fill='%233ee07a'/%3E%3C/svg%3E"/>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background: #0d1110; color: #e8eee8; }
    header { display: flex; justify-content: space-between; align-items: center; padding: 14px 22px; background: #141a17; border-bottom: 1px solid #2a362c; }
    header h1 { font-size: 18px; font-weight: 600; margin: 0; }
    .badge { font-size: 12px; padding: 4px 10px; border-radius: 999px; background: #3a2a12; color: #ffcc66; }
    .badge.ok { background: #16341f; color: #7dff9a; }
    main { display: grid; grid-template-columns: 1fr 320px; gap: 16px; padding: 16px; }
    .view { background: #111614; border: 1px solid #2a362c; border-radius: 12px; overflow: hidden; }
    .view img { width: 100%; display: block; background: #000; }
    .panel { background: #111614; border: 1px solid #2a362c; border-radius: 12px; padding: 16px; }
    .lightbar { display: flex; gap: 6px; justify-content: center; margin: 10px 0 18px; }
    .cell { width: 22px; height: 48px; border-radius: 4px; background: #1c241e; border: 1px solid #2e3b31; }
    .cell.on.left, .cell.on.right { background: #3aa0ff; box-shadow: 0 0 12px #3aa0ff88; }
    .cell.center.on { background: #3ee07a; box-shadow: 0 0 14px #3ee07a88; }
    .hint { font-size: 42px; font-weight: 700; text-align: center; margin: 8px 0 4px; letter-spacing: .08em; }
    .hint.LEFT, .hint.RIGHT { color: #7ec8ff; }
    .hint.CENTER { color: #7dff9a; }
    .hint.HOLD { color: #888; }
    .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; }
    .metric { background: #0c100e; border-radius: 8px; padding: 10px; }
    .metric .k { font-size: 11px; color: #8aa090; text-transform: uppercase; }
    .metric .v { font-size: 22px; font-variant-numeric: tabular-nums; margin-top: 4px; }
    button { background: #1d2a22; color: #e8eee8; border: 1px solid #3b4a3e; border-radius: 8px; padding: 8px 12px; cursor: pointer; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
    footer { padding: 0 22px 18px; color: #8aa090; font-size: 12px; }
  </style>
</head>
<body>
  <header>
    <h1>Orchard row guidance  ·  D455 prototype</h1>
    <div id="mode" class="badge">VISION GUIDANCE</div>
  </header>
  <main>
    <section class="view"><img id="cam" src="/video" alt="camera"/></section>
    <aside class="panel">
      <div class="hint" id="hint">HOLD</div>
      <div class="lightbar" id="bar"></div>
      <div class="metrics">
        <div class="metric"><div class="k">Lateral</div><div class="v" id="lat">—</div></div>
        <div class="metric"><div class="k">Heading</div><div class="v" id="hdg">—</div></div>
        <div class="metric"><div class="k">Speed</div><div class="v" id="spd">—</div></div>
        <div class="metric"><div class="k">Confidence</div><div class="v" id="conf">—</div></div>
        <div class="metric"><div class="k">Spacing</div><div class="v" id="spc">—</div></div>
        <div class="metric"><div class="k" id="objk">Trunks</div><div class="v" id="trk">—</div></div>
      </div>
      <div class="row">
        <button onclick="post('/api/auto', {on:true})">Auto drive</button>
        <button onclick="post('/api/auto', {on:false})">Hold heading</button>
      </div>
      <div class="row">
        <button onclick="post('/api/reset', {lateral:0.7, yaw:8})">Start offset</button>
        <button onclick="post('/api/reset', {lateral:0, yaw:0})">Start centered</button>
      </div>
    </aside>
  </main>
  <footer>Positive lateral = right of row center (steer LEFT). Speed uses tracked trunks plus planted spacing.</footer>
  <script>
    const half = 5;
    const bar = document.getElementById('bar');
    for (let i = -half; i <= half; i++) {
      const d = document.createElement('div');
      d.className = 'cell' + (i === 0 ? ' center' : '');
      d.dataset.i = i;
      bar.appendChild(d);
    }
    function fmt(v, u, d=2) { return (v===null || v===undefined) ? '—' : Number(v).toFixed(d) + u; }
    async function tick() {
      const s = await (await fetch('/api/state')).json();
      document.getElementById('hint').textContent = s.hint || 'HOLD';
      document.getElementById('hint').className = 'hint ' + (s.hint || 'HOLD');
      document.getElementById('lat').textContent = fmt(s.lateral_error_m, ' m');
      document.getElementById('hdg').textContent = fmt(s.heading_error_deg, '°', 1);
      document.getElementById('spd').textContent = fmt(s.speed_mps, ' m/s');
      document.getElementById('conf').textContent = fmt((s.confidence||0)*100, ' %', 0);
      document.getElementById('spc').textContent = fmt(s.measured_spacing_m, ' m');
      document.getElementById('trk').textContent = s.trunks ?? '—';
      document.getElementById('objk').textContent = 'Trunks';
      const lb = s.lightbar || 0;
      [...bar.children].forEach(el => {
        const i = Number(el.dataset.i);
        el.classList.remove('on','left','right');
        if (i === 0 && Math.abs(lb) < 1 && s.hint === 'CENTER') el.classList.add('on');
        if (lb > 0 && i < 0 && i >= -lb) el.classList.add('on','left');
        if (lb < 0 && i > 0 && i <= -lb) el.classList.add('on','right');
      });
      const mode = document.getElementById('mode');
      mode.textContent = (s.source === 'realsense') ? 'D455 LIVE' : 'SIMULATOR  ·  GNSS DEGRADED';
      mode.className = 'badge' + (s.confidence > 0.5 ? ' ok' : '');
    }
    async function post(url, body) {
      await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    }
    setInterval(tick, 400); tick();
  </script>
</body>
</html>
"""


def _encode_jpeg(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else b""


def _loop() -> None:
    global _jpeg, _state
    assert _pipeline is not None
    dt = 1.0 / _cfg.sim_fps
    last_print = 0.0
    while not _stop:
        t0 = time.time()
        if _sim is not None:
            _sim.step(dt)
            frame = _sim.read()
        else:
            frame = _source_obj.read()
            if frame is None:
                time.sleep(0.02)
                continue
        out = _pipeline.process(frame)
        vis = annotate(frame, _pipeline.last_perception, out, _cfg)
        jpeg = _encode_jpeg(vis)
        with _lock:
            _jpeg = jpeg
            _state = out.to_dict()
        if _sim is not None and _sim.auto_drive:
            _sim.apply_steer(out.steer)
        now = time.time()
        if _source == "realsense" or _cfg.desk_mode:
            if now - last_print >= _cfg.print_period_s:
                last_print = now
                lat = "—" if out.lateral_error_m is None else f"{out.lateral_error_m:+.2f}"
                print(
                    f"LIVE {out.hint:6}  lat={lat}  trunks={out.trunks}  conf={out.confidence:.2f}",
                    flush=True,
                )
        elapsed = now - t0
        if _sim is not None:
            time.sleep(max(0.0, dt - elapsed))


@app.get("/")
def index():
    return INDEX_HTML


@app.get("/api/state")
def api_state():
    with _lock:
        return jsonify(_state)


@app.post("/api/auto")
def api_auto():
    body = request.get_json(force=True, silent=True) or {}
    if _sim is not None:
        _sim.auto_drive = bool(body.get("on", True))
    return jsonify({"ok": True, "auto": None if _sim is None else _sim.auto_drive})


@app.post("/api/reset")
def api_reset():
    body = request.get_json(force=True, silent=True) or {}
    if _sim is not None:
        _sim.reset(float(body.get("lateral", 0.65)), float(body.get("yaw", 7.0)))
        _pipeline.reset()
    return jsonify({"ok": True})


@app.get("/video")
def video():
    def gen():
        while True:
            with _lock:
                frame = _jpeg
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(1.0 / 20.0)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main() -> None:
    global _sim, _pipeline, _source, _source_obj, _cfg
    parser = argparse.ArgumentParser(description="Orchard row guidance prototype")
    parser.add_argument("--source", choices=("sim", "realsense"), default="sim")
    parser.add_argument("--bag", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--row-width", type=float, default=None)
    parser.add_argument("--tree-spacing", type=float, default=None)
    parser.add_argument("--pens", action="store_true", help="Desk mode: steer between two pens held in front of the D455")
    parser.add_argument("--print-period", type=float, default=None, help="Seconds between readable LIVE lines (default 0.75)")
    args = parser.parse_args()
    if args.row_width:
        _cfg.row_width_m = args.row_width
    if args.tree_spacing:
        _cfg.tree_spacing_m = args.tree_spacing
    if args.pens:
        _cfg.apply_desk_pens()
    if args.print_period is not None:
        _cfg.print_period_s = max(0.15, args.print_period)
    _source = args.source
    if args.source == "sim":
        _sim = OrchardSimulator(_cfg)
        _pipeline = GuidancePipeline(_cfg, source_name="sim")
    else:
        from orchard_guidance.camera import RealSenseSource
        _source_obj = RealSenseSource(bag_path=args.bag, config=_cfg)
        _pipeline = GuidancePipeline(_cfg, source_name="realsense")
    threading.Thread(target=_loop, daemon=True).start()
    print(f"Dashboard: http://127.0.0.1:{args.port}  source={args.source}  print every {_cfg.print_period_s:.2f}s")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
