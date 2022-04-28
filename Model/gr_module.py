import torch
import torch.nn as nn
import torch.nn.functional as F
from util.knn_utils import get_neighbor_feature


# The inner relationship aggregation module
class GroupRelationAggregator(nn.Module):
    def __init__(self, in_channel, out_channel, device=torch.device('cuda:0'), num_neighbor=10, num_group=1, feature_relation_model="concat"):
        super(GroupRelationAggregator, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.device = device
        self.feature_relation_mode = feature_relation_model
        self.num_neighbor = num_neighbor
        self.num_group = num_group
        self.point_mapping_layer = nn.Conv2d(10, int(self.out_channel/16), kernel_size=1, bias=False)
        self.point_bn = nn.BatchNorm2d(int(self.out_channel/16))
        self.eta_layer = nn.Linear(self.in_channel, int(self.out_channel/16), bias=False)
        self.eta_layer_bn = nn.BatchNorm2d(int(self.out_channel/16))
        self.mu_layer = nn.Linear(self.in_channel, int(self.out_channel/16), bias=False)
        self.mu_layer_bn = nn.BatchNorm2d(int(self.out_channel/16))
        self.gamma_layer = nn.Linear(self.in_channel+3, int(self.out_channel/4), bias=False)
        self.gamma_layer_bn = nn.BatchNorm2d(int(self.out_channel/4))
        
        if self.feature_relation_mode == 'concat':
            self.mapping_layer = nn.Conv2d(int(self.out_channel*3/16), self.num_group, kernel_size=1, bias=False)
        else:
            self.mapping_layer = nn.Conv2d(int(self.out_channel/8), self.num_group, kernel_size=1, bias=False)
        
        self.bn = nn.BatchNorm2d(self.num_group)
        self.fc = nn.Linear(int(out_channel/4), out_channel, bias=False)
        
    def forward(self, features, idx=None):
        neighbor_feature = get_neighbor_feature(features, self.num_neighbor, idx, self.device)
        _, _, dimension = features.shape
        point_neighbor = neighbor_feature[:, :, :, 0:3]
        if dimension > 3:
            point_neighbor_feature = neighbor_feature[:, :, :, 3:]
            feature = features[:, :, 3:]
        else:
            point_neighbor_feature = point_neighbor
            feature = features[:, :, 0:3]
        point = features[:, :, 0:3].unsqueeze(2).repeat(1, 1, self.num_neighbor, 1)
        point = torch.cat((torch.sum((point-point_neighbor)**2, dim=3, keepdim=True), point, point_neighbor, point-point_neighbor), dim=3)
        point = F.relu(self.point_bn(self.point_mapping_layer(point.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).contiguous())
        feature = self.eta_layer(feature).unsqueeze(2).repeat(1, 1, self.num_neighbor, 1).permute(0, 3, 1, 2).contiguous()
        feature = self.eta_layer_bn(feature).permute(0, 2, 3, 1).contiguous()
        point_neighbor_feature = self.mu_layer(point_neighbor_feature).permute(0, 3, 1, 2).contiguous()
        point_neighbor_feature = self.mu_layer_bn(point_neighbor_feature).permute(0, 2, 3, 1).contiguous()
        
        if self.feature_relation_mode == 'concat':
            feature = torch.cat((feature, point_neighbor_feature), dim=3)
        elif self.feature_relation_mode == 'sum':
            feature = feature + point_neighbor_feature
        elif self.feature_relation_mode == 'sub':
            feature = feature - point_neighbor_feature
        else:
            feature = feature * point_neighbor_feature
            
        feature = torch.cat((point, feature), dim=3)
        feature = F.relu(self.bn(self.mapping_layer(feature.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).contiguous())
        neighbor_feature = self.gamma_layer(neighbor_feature).permute(0, 3, 1, 2).contiguous()
        neighbor_feature = self.gamma_layer_bn(neighbor_feature).permute(0, 2, 3, 1).contiguous()
        
        feature = feature.permute(0, 3, 1, 2).contiguous()  # [B, K, N, G]
        neighbor_feature = neighbor_feature.permute(0, 3, 1, 2).contiguous()  # [B, C, N, G]
        feature = feature.repeat(1, int(self.out_channel/4/self.num_group), 1, 1)
        out = feature * neighbor_feature
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.max(dim=2, keepdim=False)[0]
        out = self.fc(out)
        
        return out


if __name__ == '__main__':
    device = torch.device('cpu')
    in_channel = 3
    out_channel = 32
    x = torch.randn((128, 1000, 6))
    num_group = 2
    model = GroupRelationAggregator(in_channel, out_channel, device=device, num_group=num_group)
    y = model(x)
    print(y.size())
