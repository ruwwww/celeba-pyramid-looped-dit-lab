# Tasks Checklist

- [x] Implement `models/pyramid_looped_dit.py` with pure linear heads and unpatchify (no CNN upsampler), pyramid readout at 128x128, REPA projection
- [x] Implement `models/losses.py` with multi-objective perceptual supervision (LPIPS, P-DINO, REPA, Noise Gating g(t))
- [x] Implement `models/diffusion.py` and `models/__init__.py`
- [x] Implement `tests/test_pyramid_looped_dit.py` and pass all test cases
- [x] Update `train.py` to support multi-objective perceptual loss logging
- [x] Verify test suite and forward/backward memory on RTX 5060 Ti
- [x] Report verification results to user without starting full training
