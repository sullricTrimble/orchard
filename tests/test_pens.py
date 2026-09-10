import numpy as np

from orchard_guidance.config import GuidanceConfig
from orchard_guidance.perception import perceive
from orchard_guidance.pipeline import GuidancePipeline
from orchard_guidance.types import Frame, Intrinsics


def _pen_frame(left_x=-0.18, right_x=0.18, z=0.70, width=320, height=240):
    k = Intrinsics.from_fov(width, height, 87.0, 58.0)
    depth = np.zeros((height, width), np.float32)
    rgb = np.zeros((height, width, 3), np.uint8)

    def blit(xw):
        for v in range(height // 2 - 50, height // 2 + 50):
            for du in range(-4, 5):
                u = int(round(k.fx * xw / z + k.cx + du))
                if 0 <= u < width:
                    depth[v, u] = z
                    rgb[v, u] = (30, 90, 200)

    blit(left_x)
    blit(right_x)
    return Frame(rgb=rgb, depth_m=depth, intrinsics=k, timestamp_s=0.0)


def _cfg():
    return GuidanceConfig().apply_desk_pens()


def test_two_pens_centered_is_near_zero():
    perc = perceive(_pen_frame(-0.16, 0.16), _cfg())
    assert perc.lateral_error_m is not None
    assert abs(perc.lateral_error_m) < 0.04
    assert perc.trunks and len(perc.trunks) == 2


def test_two_pens_right_of_gap_steers_left():
    # Pens sit left of the camera axis → camera is right of the gap.
    perc = perceive(_pen_frame(-0.28, 0.04), _cfg())
    out = GuidancePipeline(_cfg()).process(_pen_frame(-0.28, 0.04))
    assert perc.lateral_error_m is not None and perc.lateral_error_m > 0.08
    assert out.hint == "LEFT"


def test_one_pen_holds():
    cfg = _cfg()
    frame = _pen_frame(-0.20, 0.20)
    frame.depth_m[:, frame.depth_m.shape[1] // 2 :] = 0
    perc = perceive(frame, cfg)
    assert perc.lateral_error_m is None
    assert "need_both_pens" in perc.notes
