from __future__ import annotations

from _test_helpers import outputs_for_mode


class Model:
    def infer(self, img0, img1):
        outputs = outputs_for_mode(img0, "average")
        for tensor in outputs.values():
            if tensor.device != img0.device:
                raise RuntimeError("output device mismatch")
            if tensor.dtype != img0.dtype:
                raise RuntimeError("output dtype mismatch")
        return outputs
