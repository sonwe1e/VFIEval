from __future__ import annotations

from _test_helpers import outputs_for_mode


class Model:
    def infer(self, img0, img1):
        return outputs_for_mode(img0, "img1")
