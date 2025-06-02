from module.resnet import *
from loss_fuction import *
from multi_level_visual_fusion import *


class FKEGW(nn.Module):
    def __init__(self, block, num_block, in_channels, out_channels, num_classes, img_size, fpn_cell_repeats):
        super().__init__()
        self.image = img_size
        self.weld_pool = ResNet(block=block, num_block=num_block, in_channels=in_channels,
                                out_channels=out_channels, num_classes=num_classes)
        self.multi_fusion = Multi_BiFPN(num_channels=out_channels[-1], conv_channels=out_channels,
                                        fpn_cell_repeats=fpn_cell_repeats)
        self.fc1 = nn.Linear(out_channels[-1], num_classes)
        self.fc2 = nn.Linear(num_classes, 3)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        # self.FL = focal_loss(alpha=alpha, num_classes=num_classes, gamma=gamma)

    def forward(self, grp_images, grp_labels1, grp_labels2, m, step):
        alpha1 = torch.sigmoid(self.alpha)
        beta1 = 1.0 - alpha1
        height, width = self.image, self.image
        x = grp_images.view(m, 3, height, width)
        feature1, feature2, feature3, feature4 = self.weld_pool(x)
        features = (feature1, feature2, feature3, feature4)
        feature = self.multi_fusion(features)
        output = self.fc1(feature.view(m, -1))
        output1 = self.fc2(output.view(m, -1))
        output1 = F.softmax(output1, dim=1)
        criterion1 = MyLogicalLoss()
        criterion2 = MyWeightLoss()
        loss_semantic = criterion2(output1, grp_labels2, m, 3)
        loss_classification, loss_c, loss_d, loss_e, factor = criterion1(output, grp_labels1, step)
        loss = loss_classification + alpha1 * factor * (loss_c + loss_d + loss_e) / 3 + beta1 * loss_semantic
        return output, loss, loss_classification, loss_c, loss_d, loss_e, loss_semantic