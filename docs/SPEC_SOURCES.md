# Specification sources

FewShotLab is specified by three PDF documents dated 1 November 2026. Under decision D-004 they
are kept out of git, because the repository is public. They are identified here by SHA-256 so any
copy can be checked.

| Document | File | Bytes | SHA-256 |
|---|---|---:|---|
| 01 Technical Blueprint | `FewShotLab_01_Technical_Blueprint.pdf` | 138,485 | `153949362dd5b4b3cb4d4bf5192998f4a60848132f2d3fb97d61d811747e4a38` |
| 02 Testing, Benchmarking & Judge Demo | `FewShotLab_02_Evaluation_and_Demo_Protocols.pdf` | 134,926 | `5f6db057fbedfa0481a0f2ce02e717ed00409323955dc4ae8a7d60c4eaffcdc9` |
| 03 Claude Implementation Promptbook | `FewShotLab_03_Claude_Implementation_Promptbook.pdf` | 108,829 | `72b08150d806da2878ef7e264c7532e110dd5f405626288ca69acbee094f6a77` |

## Primary references cited by the specification

The specification lists these as checked on 2026-11-01. Dataset and model facts come from these
sources; experiment settings are project decisions.

- [S1] Oxford VGG, 102 Category Flower Dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- [S2] torchvision Flowers102 source: https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/flowers102.html
- [S3] torchvision ResNet18 weights: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- [S4] K. Song and Y. Yan, NEU surface defect database: https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm
- [S5] PyTorch transfer-learning tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- [S6] Snell, Swersky and Zemel (2017), Prototypical Networks: https://arxiv.org/abs/1703.05175
- [S7] PyTorch reproducibility notes: https://docs.pytorch.org/docs/stable/notes/randomness.html
- [S8] DINOv2 official repository: https://github.com/facebookresearch/dinov2
- [S9] He et al. (2016), Deep Residual Learning: https://arxiv.org/abs/1512.03385
- [S10] scikit-learn classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
- [S11] TensorFlow Datasets, oxford_flowers102: https://www.tensorflow.org/datasets/catalog/oxford_flowers102
- [S12] Guo et al. (2017), On Calibration of Modern Neural Networks: https://arxiv.org/abs/1706.04599
