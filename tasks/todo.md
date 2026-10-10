# Tasks Checklist

- [ ] Implement `models/pyramid_looped_dit.py` (~50M param pixel-space transformer, bottleneck patch embed, QK-Norm, looped core blocks, adaLN)
- [ ] Implement `models/diffusion.py` (JiT x-prediction flow loss, v-weighted MSE, Euler ODE sampler with CFG)
- [ ] Implement `models/__init__.py`
- [ ] Implement `tests/test_pyramid_looped_dit.py` and pass all unit tests
- [ ] Implement `train.py` with screening mode and W&B tracking
- [ ] Implement `sample.py` for direct RGB image generation
