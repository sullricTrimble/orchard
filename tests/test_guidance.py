from orchard_guidance.config import GuidanceConfig
from orchard_guidance.pipeline import GuidancePipeline
from orchard_guidance.simulator import OrchardSimulator


def _cfg():
    cfg = GuidanceConfig()
    cfg.sim_spacing_jitter_m = 0.02
    cfg.sim_lateral_jitter_m = 0.02
    cfg.image_width = 640
    cfg.image_height = 360
    cfg.sim_fps = 15.0
    return cfg


def test_centered_is_near_zero():
    cfg = _cfg()
    sim = OrchardSimulator(cfg, seed=1)
    sim.reset(lateral_m=0.0, yaw_deg=0.0, s_m=8.0)
    out = GuidancePipeline(cfg).process(sim.read())
    assert out.lateral_error_m is not None and abs(out.lateral_error_m) < 0.20
    assert out.heading_error_deg is not None and abs(out.heading_error_deg) < 3.5


def test_right_offset_steers_left():
    cfg = _cfg()
    sim = OrchardSimulator(cfg, seed=1)
    sim.reset(lateral_m=0.70, yaw_deg=0.0, s_m=8.0)
    out = GuidancePipeline(cfg).process(sim.read())
    assert out.lateral_error_m is not None and out.lateral_error_m > 0.35
    assert out.steer > 0.0 and out.hint == "LEFT"


def test_left_offset_steers_right():
    cfg = _cfg()
    sim = OrchardSimulator(cfg, seed=1)
    sim.reset(lateral_m=-0.70, yaw_deg=0.0, s_m=8.0)
    out = GuidancePipeline(cfg).process(sim.read())
    assert out.lateral_error_m is not None and out.lateral_error_m < -0.35
    assert out.steer < 0.0 and out.hint == "RIGHT"


def test_yaw_right_reports_positive_heading():
    cfg = _cfg()
    sim = OrchardSimulator(cfg, seed=1)
    sim.reset(lateral_m=0.0, yaw_deg=8.0, s_m=8.0)
    out = GuidancePipeline(cfg).process(sim.read())
    assert out.heading_error_deg is not None and out.heading_error_deg > 3.0


def test_closed_loop_recenters():
    cfg = _cfg()
    sim = OrchardSimulator(cfg, seed=2)
    sim.reset(lateral_m=0.75, yaw_deg=6.0, s_m=6.0)
    sim.auto_drive = True
    pipe = GuidancePipeline(cfg)
    dt = 1.0 / cfg.sim_fps
    last = None
    for _ in range(int(12 * cfg.sim_fps)):
        sim.step(dt)
        last = pipe.process(sim.read())
        sim.apply_steer(last.steer)
    assert last is not None and last.lateral_error_m is not None
    assert abs(last.lateral_error_m) < 0.30
    assert abs(sim.pose.x_m) < 0.35


def test_speed_from_tree_motion():
    cfg = _cfg()
    cfg.sim_speed_mps = 2.0
    sim = OrchardSimulator(cfg, seed=4)
    sim.reset(lateral_m=0.05, yaw_deg=0.0, s_m=6.0)
    sim.pose.speed_mps = 2.0
    sim.auto_drive = False
    pipe = GuidancePipeline(cfg)
    dt = 1.0 / cfg.sim_fps
    last = None
    for _ in range(int(3.0 * cfg.sim_fps)):
        sim.step(dt)
        last = pipe.process(sim.read())
    assert last is not None and last.speed_mps is not None
    assert 1.2 < last.speed_mps < 2.8
    assert last.trunks >= 2
