# Measuring the Contribution of Multiple Models in Detecting Adversarial Instances

This repository contains the code for *Measuring the Contribution of Multiple Model Representations
in Detecting Adversarial Instances*.

<div align="center">
 <img src="https://github.com/dstein64/media/blob/main/multi-adv-detect/illustration.svg?raw=true" width="560"/>
</div>

Reported running times are approximate, intended to give a general idea of how long each step will
take. Estimates are based on times encountered while developing on Ubuntu 21.04 with hardware that
includes an AMD Ryzen 9 3950X CPU, 64GB of memory, and an NVIDIA TITAN RTX GPU with 24GB of memory.
The intermediate results utilize about 600 gigabytes of storage.

### Requirements

The code was developed using Python 3.9 on Ubuntu 21.04. Other systems and Python versions may work,
but have not been tested.

Python library dependencies are specified in [requirements.txt](requirements.txt). Versions are
pinned for reproducibility.

### Installation

- Optionally create and activate a virtual environment.

```shell
python3 -m venv env
source env/bin/activate
```

- Install Python dependencies, specified in `requirements.txt`.
  * 2 minutes

```shell
pip3 install -r requirements.txt
```

### Running the Code

By default, output is saved to the `./workspace` directory, which is created automatically.

- Train ResNet classification models.
  * 6 weeks

```shell
python3 src/train_nets.py
```

- Evaluate the models, extracting representations from the corresponding data.
  * 1 hour

```shell
python3 src/eval_nets.py
```

- Adversarially perturb test images, evaluating and extracting representations from the
  corresponding data.
  * 21 hours

```shell
python3 src/attack.py
```

- Train and evaluate model-wise control adversarial instance detectors, varying the number of
  underlying models used for generating features, where the underlying detectors are trained on
  representations from a single model.
  * 1 day

```shell
OMP_NUM_THREADS=1 python3 src/detect_model_wise_control.py
```

- Train and evaluate model-wise treatment adversarial instance detectors, varying the number of
  underlying models used for generating features, where the underlying detectors are trained on
  representations from multiple models.
  * 1 day

```shell
OMP_NUM_THREADS=1 python3 src/detect_model_wise_treatment.py
```

- Train and evaluate unit-wise control adversarial instance detectors, varying the number of units
  used for generating features, where the units come from a single underlying model.
  * 1 hour

```shell
OMP_NUM_THREADS=1 python3 src/detect_unit_wise_control.py
```

- Train and evaluate unit-wise treatment adversarial instance detectors, varying the number of units
  used for generating features, where the units come from multiple underlying models.
  * 2 hours

```shell
OMP_NUM_THREADS=1 python3 src/detect_unit_wise_treatment.py
```

- Generate plots.
  * 2 seconds

```shell
python3 src/plot.py
```
