# MAGI-X

This repository provides the PyTorch implementation of MAGI-X, an extension of the MAnifold-constrained Gaussian process Inference ([MAGI][1]) for learning unknown ODEs system, where the associated derivative function does not have known parametric form. 

### MAGI-X codes and requirements
The codes for MAGI-X are provided under directory ```magix/```. To successfully run the MAGI-X code, we require the installation of the following python packages. We provide the version that we use, but other version of the packages is also allowed as long as it is compatible.

```sh
pip3 install numpy==1.19.5 scipy==1.6.1 torch==1.8.0 matplotlib==3.3.4
```

Please see ```demo.ipynb``` for tutorial of running MAGI-X.

### References

Our paper is available on [arXiv][2]. If you found this repository useful in your research, please consider citing

```
@article{huang2021magi-x,
  title={MAGI-X: Manifold-Constrained Gaussian Process Inference for Unknown System Dynamics},
  author={Huang, Chaofan and Ma, Simin and Yang, Shihao},
  journal={arXiv preprint arXiv:2105.12894},
  year={2021}
}
```

[1]: https://www.pnas.org/content/118/15/e2020397118.short
[2]: https://arxiv.org/abs/2105.12894