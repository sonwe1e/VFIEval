from __future__ import annotations

import torch.nn.functional as F

from _test_helpers import outputs_for_mode


class Model:
    def infer(self, img0, img1):
        outputs = outputs_for_mode(img0, "average")
        return {
            "flowt_0": F.interpolate(outputs["flowt_0"], size=(4, 4), mode="bilinear", align_corners=True),
            "flowt_1": F.interpolate(outputs["flowt_1"], size=(4, 4), mode="bilinear", align_corners=True),
            "mask0": F.interpolate(outputs["mask0"], size=(4, 4), mode="bilinear", align_corners=True),
            "mask1": F.interpolate(outputs["mask1"], size=(4, 4), mode="bilinear", align_corners=True),
        }
