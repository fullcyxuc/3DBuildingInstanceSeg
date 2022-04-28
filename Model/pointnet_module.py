"""
Created on 2021/11/22 下午3:23

@author: xue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
	def __init__(self, in_channel, out_channel, m):
		super(PointNetEncoder, self).__init__()
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.m = m

		self.conv1 = nn.Conv1d(self.in_channel, self.m * 2, 1)
		self.conv2 = nn.Conv1d(self.m * 2, self.m * 4, 1)
		self.conv3 = nn.Conv1d(self.m * 4, self.m * 8, 1)
		self.conv4 = nn.Conv1d(self.m * 8 + self.m * 2, out_channel, 1)

		self.bn1 = nn.BatchNorm1d(self.m * 2)
		self.bn2 = nn.BatchNorm1d(self.m * 4)
		self.bn3 = nn.BatchNorm1d(self.m * 8)
		self.bn4 = nn.BatchNorm1d(self.out_channel)

	def forward(self, xyz, feats):
		x = torch.cat([xyz, feats], dim=-1).permute(0, 2, 1).contiguous()  # [B, 3+f_dim, N]
		B, D, N = x.size()

		x = F.relu(self.bn1(self.conv1(x)))  # [B, m*2, N]
		pointfeat = x
		x = F.relu(self.bn2(self.conv2(x)))  # [B, m*4, N]
		x = F.relu(self.bn3(self.conv3(x)))  # [B, m*8, N]

		x = torch.max(x, 2, keepdim=True)[0]  # [B, m*8, 1]
		x = x.view(-1, self.m * 8, 1).repeat(1, 1, N)  # [B, m*8, N]
		x = torch.cat([x, pointfeat], 1)  # [B, m*8 + m*2, N]

		x = self.bn4(self.conv4(x))  # [B, out_dim, N]
		x = x.permute(0, 2, 1).contiguous()
		return x