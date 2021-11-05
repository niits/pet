import torch
from torch import nn
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame


class PetPopularityRegression(nn.Module):
    def __init__(
        self,
        layers: list = [3, 4, 5, 7],
        global_pool: str = "",
        inchans: int = 3,
        preact: bool = False,
        stem_type: str = "same",
        attribute_length: int = 12,
        input_shape: int = 256,
        init: bool = False,
    ) -> None:
        super(PetPopularityRegression, self).__init__()
        self.backbone = ResNetV2(
            layers=layers,
            global_pool=global_pool,
            in_chans=inchans,
            preact=preact,
            stem_type=stem_type,
            conv_layer=StdConv2dSame,
        )

        self.linear = nn.Linear(
            in_features=int((input_shape / 2 ** 5) ** 2 * 1000 + attribute_length),
            out_features=1,
        )

        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform(self.linear.weight)

    def forward(self, x):
        image_tensor, attribute_tensor = x

        feature_map = self.backbone(image_tensor)

        regress_output = self.linear(
            torch.cat(
                (
                    torch.flatten(feature_map, start_dim=1),
                    attribute_tensor,
                ),
                dim=1,
            )
        )
        return self.sigmoid(regress_output) * 100
