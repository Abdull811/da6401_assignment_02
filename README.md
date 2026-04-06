# DA6401 Assignment 02 — Multi-Task Perception Model

## Overview

This project implements a **multi-task deep learning system** on the **Oxford-IIIT Pet Dataset**, solving three tasks simultaneously:

* **Classification** → Predict pet breed (37 classes)
* **Localization** → Predict bounding box
* **Segmentation** → Pixel-wise mask prediction

The system uses a **VGG11-based architecture** with task-specific heads and includes full **training, evaluation, visualization, and inference pipeline**.

## Repository
* GitHub Repository: [https://github.com/Abdull811/DA6401_assigment_01.git]
* Weights & Biases Run Link: View Run on W&B [https://wandb.ai/ge26z811-zan/da6401_Assigment_02_Weight_&_Biase?nw=nwuserge26z811]


---

## Project Structure

```
da6401_assignment_02/
│
├── checkpoints
│   └── checkpoints.md
│
├── data/
│   └── pets_dataset.py
│
├── models/
│   ├── vgg11.py
│   ├── classification.py
│   ├── localization.py
│   ├── segmentation.py
│   ├── multitask.py
│   └── layers.py
│
├── losses/
│   ├── iou_loss.py
│   └── __init__.py
│
├── train.py
├── inference.py
├── requirements.txt
├── README.md
└── *.pth (generated after training)
```

---

## Installation

```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Download Oxford-IIIT Pet dataset:

 https://www.robots.ox.ac.uk/~vgg/data/pets/

Extract into:

```
data/
├── images/
├── annotations/
│   ├── trimaps/
│   ├── trainval.txt
│   └── test.txt
```

---

## Training

Run training:

```bash
python train.py
```

### Training includes:

* Classification loss (CrossEntropy)
* Localization loss (MSE + IoU)
* Segmentation loss (CrossEntropy + Dice)
* W&B logging (metrics + images)

---

## Weights & Biases (W&B)

Login before running:

```bash
wandb login
```

Project name:

```
da6401_Assigment_02_Weight_&_Biase
```

### Logged metrics:

* Classification loss
* Localization loss
* Segmentation loss
* Dice score
* Pixel accuracy
* IoU

### Logged visuals:

* Input images
* Ground truth masks
* Predicted masks
* Feature maps
* IoU table

---

## Model Architecture

### Backbone

* VGG11 Encoder

### Heads

* Classification Head → Fully connected layers
* Localization Head → Bounding box regression
* Segmentation Head → U-Net decoder

---

## Loss Functions

| Task           | Loss                     |
| -------------- | ------------------------ |
| Classification | CrossEntropy             |
| Localization   | MSE + IoU Loss           |
| Segmentation   | CrossEntropy + Dice Loss |

---

## Metrics

* Dice Score
* Pixel Accuracy
* IoU (Bounding Box)

---

## Inference

Run:

```bash
python inference.py
```

### Output:

* Predicted segmentation mask
* Logged to W&B

---

## Checkpoints

After training, the following files are generated:

```
classifier.pth
localizer.pth
unet.pth
```

These are required for:

* Evaluation
* Multi-task model initialization

---

## Multi-Task Model

The `MultiTaskPerceptionModel` combines:

* Classification
* Localization
* Segmentation

into a single forward pass:

```python
output = model(image)

output["classification"]
output["localization"]
output["segmentation"]
```

---



## Notes

* Ensure dataset path is correct (`data/`)
* Do NOT include `.pth` files in GitHub (large files)
* Use `.gitignore` for:

  ```
  data/
  *.pth
  wandb/
  ```

## Conclusion

This project demonstrates a complete **multi-task deep learning pipeline**, integrating:

* Classification
* Localization
* Segmentation

with proper training, evaluation, and visualization tools.

---
