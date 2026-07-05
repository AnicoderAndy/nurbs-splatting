# NURBS Splatting

We introduce NURBS Splatting, a differentiable rendering framework that represents planar rational splines as Gaussian fields, enabling stable optimization and unified support for rendering, filling, and vector graphics applications. [[paper](https://arxiv.org/abs/2606.31764)]

<details>
<summary>Click here for the Full Abstract</summary>
Differentiable rendering of planar rational splines remains largely underexplored, despite their widespread use in vector graphics and design. Existing differentiable vector renderers primarily focus on Bézier curves and rely on analytic rasterization, which can suffer from gradient instability and limited flexibility. We propose NURBS Splatting, a unified framework that represents planar rational curves as continuous Gaussian fields. By sampling Gaussians along the curve parameter domain and inside closed regions, rendering is reformulated as a smooth accumulation process with stable gradients. Our method naturally supports long splines, rational weights, non-uniform knots, and closed-region filling. We demonstrate its effectiveness in calligraphy reconstruction, vectorization frameworks, and long-spline image abstraction, showing improved stability and reconstruction quality over existing approaches.
</details>

## Installation & Usage
We build this project using Python 3.12 with a mamba environment. We highly recommend using mamba for managing dependencies. Create the environment, install [PyTorch](https://pytorch.org/), and the required dependencies.

```bash
git submodule update --init --recursive
mamba create -n nsplat python=3.12 -y
mamba activate nsplat
# Find a suitable PyTorch version if needed
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
cd external/gsplat
pip install -e ".[dev]" --no-build-isolation
cd ../..  # back to the root directory
pip install -e . --no-build-isolation
```

The calligraphy reconstruction experiments require an additional dependency [SLDVec](https://github.com/tanguymagne/SLD-Vectorization), which is included in `external/SLDVec`. Please follow the instructions in that repository to install it.

After installation, you can run the demo scripts in the `apps` folder.

## Acknowledgements
We thank Xingtong Ge et al. for their [gsplat](https://github.com/XingtongGe/gsplat/) backbone; we thank Daniel Berio et al. for their [calligraph](https://github.com/colormotor/calligraph) project which inspired our work; we also thank Tanguy Magne and Olga Sorkine-Hornung for their [SLDVec](https://github.com/tanguymagne/SLD-Vectorization) project which is essential in our calligraphy reconstruction experiments. I as the first author specially thank Yanze Ye for implementing a beautiful frontend for our project.

## Citation
If you find our work useful, please consider citing our paper:

```
@inproceedings{qiu2026nurbs,
  title={{{NURBS Splatting}}: A Unified Differentiable Rendering Framework for Vector Graphics},
  author={Qiu, Jingye and Zhou, Shizhe},
  booktitle={Computer Vision -- ECCV 2026},
  year={2026}
}
```