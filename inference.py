"""Inference and evaluation
"""

import torch
from models.multitask import MultiTaskPerceptionModel


def main():
    model = MultiTaskPerceptionModel()
    model.eval()

    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    print(output["classification"].shape)
    print(output["localization"].shape)
    print(output["segmentation"].shape)


if __name__ == "__main__":
    main()
