# SCA_Uncertainty_Decomposition
This repository provides the implementation of the framework introduced in paper "Uncertainty Estimation in Neural Network-enabled Side-channel Analysis and Links to Explainability"

This paper introduces a unified framework for quantifying and explaining predictive uncertainty in neural network–based side-channel analysis. By leveraging matrix-based Rényi entropy and α-divergence, we decompose uncertainty into epistemic and aleatoric components and analyze how data quality, physical effects, and training choices impact key recovery. To localize the sources of uncertainty in side-channel traces, we integrate Shapley value–based explanations that identify time samples most responsible for unreliable predictions. Extensive experiments show that the proposed uncertainty measures strongly correlate with standard SCA metrics such as key rank, providing a complementary lens for evaluating attack effectiveness and complexity.
# Dependencies
Install dependencies: tensorflow:
```bash
pip install tensorflow
```

scipy:
```python
pip install scipy
```

h5py:
```bash
pip install h5py
```

## Acknowledgements

1. This project uses code from the [MRE](https://github.com/SJYuCNEL/Matrix-based-Dependence/blob/main/MI.py) repository by Shujian Yu for implementation of the MRE.
2. The codes from [AutoSCA](https://github.com/AISyLab/AutoSCA/tree/main) were used to train the models.
3. [TCHES20V3_CNN_SCAPublic](https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA) and [Methodology-for-efficient-CNN-architectures-in-SCA](https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA) were used for the model trained on ASCADf dataset.
