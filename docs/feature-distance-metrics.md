# LPIPS feature-distance adapters

VFIEval keeps the public metric names `lpips_vit_patch` and
`lpips_convnext` for compatibility. Their implementations are explicitly
reported as DINOv2 patch-feature distance and ConvNeXt multi-scale feature
distance. Lower values are better; VFIEval never substitutes another score
when either implementation is unavailable.

## Asset integrity

Each feature manifest must identify the checkpoint source and pin the SHA-256
of the installed weights. The DINOv2 manifest also pins the content hash of
its local evaluator checkout. A floating download URL such as a `main` branch
is therefore not the metric identity: the installed, manifest-pinned bytes
are.

Run the existing setup command to add fingerprints to installed legacy
assets, or use `--force` to download fresh declared assets:

```text
python -m vfieval.cli --workspace .vfieval prepare-metrics
python -m vfieval.cli --workspace .vfieval prepare-metrics --force
```

Changing the manifest, adapter code, evaluator checkout, or weights changes
the metric cache identity.

## Strict load and conformance

Before a score can be published, the adapter requires every declared model
parameter name and shape to match the checkpoint. The load report records the
matched-key count and fingerprint plus all missing, unexpected, and
shape-mismatched keys. Zero-match and partial loads are `unavailable`; they
cannot produce a score.

After loading, each adapter runs a small deterministic smoke check. Identical
inputs must have distance near zero, and a perturbed input must produce a
finite, non-negative distance. Failures remain `unavailable` with the load and
conformance reports in metric result details. Metric health exposes the
manifest, adapter/evaluator, weights, and combined implementation
fingerprints, together with the latest strict-load validation for that exact
implementation fingerprint.
