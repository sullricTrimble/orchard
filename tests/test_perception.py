from orchard_guidance.config import GuidanceConfig
from orchard_guidance.perception import perceive
from orchard_guidance.simulator import OrchardSimulator


def test_detects_both_rows():
    cfg = GuidanceConfig()
    cfg.sim_spacing_jitter_m = 0.01
    cfg.sim_lateral_jitter_m = 0.01
    sim = OrchardSimulator(cfg, seed=0)
    sim.reset(0.0, 0.0, s_m=8.0)
    perc = perceive(sim.read(), cfg)
    assert perc.left_line is not None and perc.right_line is not None
    assert perc.centerline is not None
    assert perc.row_width_m is not None
    assert abs(perc.row_width_m - cfg.row_width_m) < 1.0
    assert len(perc.trunks) >= 4
