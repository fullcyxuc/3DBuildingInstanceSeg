import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model.gr_module import GroupRelationAggregator
from Model.pointnet_module import PointNetEncoder
from OP import pointnet2_utils


class RelationNet(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(RelationNet, self).__init__()
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.conv1 = nn.Sequential(nn.Conv2d(2 * in_channel, in_channel, kernel_size=1, bias=False),
								   nn.BatchNorm2d(in_channel),
								   nn.LeakyReLU(negative_slope=0.2))
		self.conv2 = nn.Sequential(nn.Conv2d(in_channel, 2 * in_channel, kernel_size=1, bias=False),
								   nn.BatchNorm2d(2 * in_channel),
								   nn.LeakyReLU(negative_slope=0.2))
		self.conv3 = nn.Sequential(nn.Conv2d(2 * in_channel, 2 * in_channel, kernel_size=1, bias=False),
								   nn.BatchNorm2d(2 * in_channel),
								   nn.LeakyReLU(negative_slope=0.2))
		self.fc1 = nn.Linear(2 * in_channel, 128)
		self.bn1 = nn.BatchNorm1d(128)
		self.fc2 = nn.Linear(128, self.out_channel)
		self.bn2 = nn.BatchNorm1d(self.out_channel)

	def forward(self, point_feature, center_feature):
		batch_size, num_point, _ = point_feature.shape
		_, num_center, _ = center_feature.shape
		point_feature = point_feature.unsqueeze(2).repeat((1, 1, num_center, 1))
		center_feature = center_feature.unsqueeze(1).repeat((1, num_point, 1, 1))
		feature = torch.cat((point_feature, center_feature), dim=3)
		feature = feature.permute(0, 3, 1, 2).contiguous()
		feature = self.conv1(feature)
		feature = self.conv2(feature)
		feature = self.conv3(feature)
		feature = feature.max(dim=-1, keepdim=False)[0].permute(0, 2, 1)  # todo
		feature = self.fc1(feature).permute(0, 2, 1).contiguous()
		feature = F.relu(self.bn1(feature)).permute(0, 2, 1).contiguous()
		feature = self.fc2(feature).permute(0, 2, 1).contiguous()
		feature = F.relu(self.bn2(feature)).permute(0, 2, 1).contiguous()

		return feature


class ScoreNet(nn.Module):
	def __init__(self, in_channel, out_channel):
		super().__init__()
		self.conv1 = nn.Conv1d(in_channel, in_channel // 2, 1)
		self.conv2 = nn.Conv1d(in_channel // 2, in_channel // 4, 1)
		self.conv3 = nn.Conv1d(in_channel // 4, out_channel, 1)

		self.bn1 = nn.BatchNorm1d(in_channel // 2)
		self.bn2 = nn.BatchNorm1d(in_channel // 4)
		self.bn3 = nn.BatchNorm1d(out_channel)

	def forward(self, xyz, feats):
		if feats is None:
			x = torch.cat((xyz, xyz), dim=2)
		else:
			x = torch.cat((xyz, feats), dim=2)
		x = x.permute(0, 2, 1).contiguous()

		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = self.bn3(self.conv3(x))

		return x.permute(0, 2, 1).contiguous()


class FeatureExtraction(nn.Module):
	def __init__(self, device, m):
		super(FeatureExtraction, self).__init__()
		self.device = device
		self.GRA1 = GroupRelationAggregator(3, m, device=self.device)
		self.GRA2 = GroupRelationAggregator(m, m * 2, device=self.device)
		self.GRA3 = GroupRelationAggregator(m * 2, m * 4, device=self.device)
		self.GRA4 = GroupRelationAggregator(m * 4, m * 8, device=self.device)

	def forward(self, xyz, feats):
		if feats is None:
			feature = torch.cat((xyz, xyz), dim=2)
		else:
			feature = torch.cat((xyz, feats), dim=2)
		feature = self.GRA1(feature)
		feature = torch.cat((xyz, feature), dim=2)
		feature = self.GRA2(feature)
		feature = torch.cat((xyz, feature), dim=2)
		feature = self.GRA3(feature)
		feature = torch.cat((xyz, feature), dim=2)
		feature = self.GRA4(feature)

		return feature


class InstanceSegPipline(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.m = cfg.m
		self.device = cfg.device
		self.prepare_epochs = cfg.prepare_epochs
		self.num_classes = cfg.classes
		self.cfg = cfg

		# backbone
		# self.backbone = FeatureExtraction(self.device, self.m)
		in_channel = 0
		if cfg.using_xyz:
			in_channel += 3
		if cfg.using_rgb:
			in_channel += 3
		if cfg.using_normal:
			in_channel += 3

		self.backbone = PointNetEncoder(in_channel, cfg.out_channel, self.m)

		# semantic segmentation
		self.semantic_linear = nn.Linear(cfg.out_channel, self.num_classes)  # bias(default): True

		# attentive map
		self.relation_net = RelationNet(cfg.out_channel, self.m)

		# scorenet
		self.score_net = ScoreNet(cfg.out_channel + 3, self.m)
		self.score_linear = nn.Linear(self.m, 1)

	def select(self, xyz, feats, candidates_num):
		"""
		FPS and then select top k candidate by considering feature entropy
		:param xyz: input xyz of each points [B, N, 3]
		:param feats: input feature of each points [B, N, F]
		:return: index of instance candidate points
		"""

		def cal_entropy(feats):
			"""
			:param feats: [B, N, F]
			:return:
			"""
			softmax_feat = F.softmax(feats, dim=-1)
			feat_entropy = torch.sum(-softmax_feat * torch.log(softmax_feat), dim=-1)  # sigma{-fi * log{fi}}, [B, N]
			return feat_entropy

		assert xyz.size()[1] == feats.size()[1]
		points_num = xyz.size()[1]

		idx = pointnet2_utils.furthest_point_sample(xyz, points_num // 2)  # fps采样，返回index [B, NSample]
		gather_feature = pointnet2_utils.gather_operation(feats.permute(0, 2, 1).contiguous(),
														  idx).permute(0, 2, 1).contiguous()  # 根据index获取采样点
		gather_feature = gather_feature.cpu()

		feat_entropy = cal_entropy(gather_feature)
		sorted_idx = feat_entropy.sort(dim=-1)[1][:, :candidates_num].long()  # [B, NSample]
		sorted_idx = sorted_idx.to(self.device)
		candidate_idx = torch.gather(idx, -1, sorted_idx)

		return candidate_idx

	def forward(self, xyz, feats):
		ret = {}
		batch_size, num_point, _ = xyz.size()

		# backbone
		output_feats = self.backbone(xyz, feats)  # [B, N, F]
		ret['embedding_feats'] = output_feats

		# semantice part
		semantic_scores = self.semantic_linear(output_feats)  # [B, N, nclass]
		semantic_preds = semantic_scores.max(dim=-1)[1]  # [B, N]

		ret['semantic_scores'] = semantic_scores.view(-1, self.cfg.classes)

		# select module
		candidates_num = xyz.size()[
							 1] // self.cfg.candidate_scale  # the number of candidates instance is the 1% of the origin points'
		candidate_idx = self.select(xyz, feats, candidates_num)  # [B, Nc]
		candidate_feats = pointnet2_utils.gather_operation(
			output_feats.permute(0, 2, 1).contiguous(), candidate_idx
		).permute(0, 2, 1).contiguous()  # [B, Nc, F]

		# relation matrix
		relation_matrix = F.softmax(self.relation_net(output_feats, candidate_feats), dim=-1)  # [B, N, Nc]
		proposals_idx = torch.argmax(relation_matrix, dim=-1)  # [B, N]
		nProposal = torch.unique(proposals_idx).size()[0]

		# score net
		score = self.score_net(xyz, output_feats)  # [B, N, F']
		score = pointnet2_utils.roipool(score, proposals_idx.int(), nProposal)  # [B, Nc, F']
		proposal_score = self.score_linear(score)  # [B, Nc, 1]

		ret['proposal_scores'] = (proposals_idx, proposal_score, nProposal)

		return ret


def model_fn_decorator(test=False):
	from util.config import cfg

	semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).to(cfg.device)
	score_criterion = nn.BCELoss(reduction='none').to(cfg.device)

	def test_model_fn(batch, model, epoch):
		pass

	def train_model_fn(batch, model, epoch):
		'''
		:param batch: input batch, contains point coordinate, feature(rgb, normal, etc), seg_label, ins_label
		:param model: model to process the data
		:param epoch: retain
		:return: loss: the total loss
		'''

		# unpack batch data										in case B = 1
		xyz = batch['locs_float'].to(cfg.device)  # [B, N, 3] point coordinate
		feats = batch['feats'].to(cfg.device)  # [B, N, 3] color
		seg_labels = batch['labels'].to(cfg.device)  # [B, N] segmentation label
		ins_labels = batch['instance_labels'].to(cfg.device)  # [B, N] instance label
		instance_num = batch['instance_num'].to(cfg.device)  # [B,] number of instance for each batch
		print('xyz size:', xyz.size(), 'instance_num:', instance_num)

		# get and unpack model result
		ret = model(xyz, feats)

		embedding_feats = ret['embedding_feats']  # [B, N, F] embedding feats for calculating the embedding loss
		semantic_scores = ret['semantic_scores']  # [B, nclass, N]
		proposals_idx, proposal_scores, proposals_num = ret['proposal_scores']  # [B, Nc, 1]

		# for loss calculation
		loss_items = {}
		loss_items['embedding_feats'] = embedding_feats
		loss_items['semantic_scores'] = (semantic_scores, seg_labels)
		loss_items['proposal_scores'] = (proposals_idx, proposal_scores, proposals_num, ins_labels, instance_num)

		loss, loss_out = loss_fn(loss_items)

		return loss

	def loss_fn(loss_item):
		'''
		:param loss_item: contain the items needed for calculating the losses
		:return:
		'''
		loss_out = {}
		## semantic loss
		semantic_scores, seg_labels = loss_item['semantic_scores']
		semantic_loss = semantic_criterion(semantic_scores, seg_labels.view(-1))
		loss_out['semantic_loss'] = semantic_loss

		## score loss
		proposals_idx, proposal_scores, proposals_num, ins_labels, instance_num = loss_item['proposal_scores']
		score_loss = 0
		if instance_num > 0:  # in case some scenes don't contain instance
			# get iou
			ious = pointnet2_utils.get_iou(proposals_idx.int(), proposals_num, ins_labels.int(),
										   instance_num)  # [B, nProposal, nInstance]
			gt_ious, _ = ious.max(dim=-1)  # [B, nProposal]

			# convert iou to binary gt
			mask_low = gt_ious < cfg.bg_thresh  # ious < low threshold
			mask_mid = (gt_ious >= cfg.bg_thresh) & (
						gt_ious <= cfg.fg_thresh)  # low threshold <= ious <= high threshold
			mask_high = gt_ious > cfg.fg_thresh  # ious > high threshold

			gt_score1 = torch.zeros_like(mask_low).float()

			k = torch.ones_like(gt_ious) * (1 / (cfg.fg_thresh - cfg.bg_thresh))
			b = torch.ones_like(gt_ious) * (cfg.bg_thresh / (cfg.bg_thresh - cfg.fg_thresh))
			gt_score2 = torch.where(mask_mid, gt_ious * torch.ones_like(gt_ious) * k + b, torch.zeros_like(gt_ious))

			gt_score3 = torch.where(mask_high, torch.ones_like(gt_ious), torch.zeros_like(gt_ious))

			gt_scores = gt_score1 + gt_score2 + gt_score3  # [B, nProposal]

			score_loss = score_criterion(torch.sigmoid(proposal_scores).view(-1), gt_scores.view(-1))
			score_loss = score_loss.mean()
		loss_out['score_loss'] = score_loss

		## embedding loss
		embedding_feats = loss_item['embedding_feats']  # [B, N, F]

		# pull loss			in only case 1 batch
		pull_loss = 0.0
		for i in range(instance_num[0]):
			instance_feats_i = embedding_feats[ins_labels == i]  # [instance_i_pnum, F]
			instance_pnum_i = instance_feats_i.size()[0]
			mu_feat_i = torch.sum(instance_feats_i, dim=0) / instance_pnum_i  # [F] average feature of the ith instance

			difference_i = instance_feats_i - mu_feat_i  # [instance_i_pnum, F]
			distance_i = torch.sum(difference_i ** 2, dim=-1)  # [instance_i_pnum]
			pull_loss_i = distance_i - cfg.sigma_1
			pull_loss_i = torch.where(pull_loss_i >= 0, pull_loss_i,
									  torch.zeros_like(pull_loss_i))  # max(0, ||mu - fi||2 - sigma1)
			pull_loss_i = torch.sum(pull_loss_i) / instance_pnum_i

			pull_loss += pull_loss_i
		pull_loss /= instance_num[0]

		# push loss
		push_loss = 0.0
		for i in range(instance_num[0]):
			for j in range(instance_num[0]):
				if i != j:
					instance_feats_i = embedding_feats[ins_labels == i]  # [instance_i_pnum, F]
					instance_feats_j = embedding_feats[ins_labels == j]  # [instance_j_pnum, F]

					mu_feat_i = torch.sum(instance_feats_i, dim=0) / instance_feats_i.size()[0]
					mu_feat_j = torch.sum(instance_feats_j, dim=0) / instance_feats_j.size()[0]

					difference_ij = mu_feat_i - mu_feat_j
					distance_ij = torch.sum(difference_ij ** 2)
					push_loss_ij = max(distance_ij, 0)
					push_loss += push_loss_ij
		push_loss /= (instance_num[0] * (instance_num[0] - 1))

		embedding_loss = pull_loss + push_loss

		loss_out['embedding_loss'] = embedding_loss

		## total loss
		loss = semantic_loss + score_loss + embedding_loss

		return loss, loss_out

	if test:
		fn = test_model_fn
	else:
		fn = train_model_fn
	return fn


if __name__ == '__main__':
	# for test roi pooling
	# [2, 3, 4]
	# feats = torch.tensor([[[1, 2, 3, 4], [2, 3, 4, 1], [3, 1, 5, 10]], [[1, 2, 3, 4], [2, 3, 4, 1], [3, 1, 5, 10]]]).float()
	# feats.requires_grad_(True)
	# feats = feats.cuda()
	# # [2, 3]
	# inst_label = torch.tensor([[0, 1, 1], [0, 0, 1]]).int()
	# inst_label = inst_label.cuda()
	feats = torch.rand((1, 8000, 32), requires_grad=True).cuda()
	inst_label = torch.zeros((1, 8000)).int().cuda()
	inst_label[:, :3000] = 1
	print(feats.size(), inst_label.size())
	res = pointnet2_utils.roipool(feats, inst_label, 2)
	print(res.size(), res)
# out = torch.mean(res)
# torch.autograd.backward(out)
# print(res.size(), res, res.grad)
# print(out.size(), out, out.grad)

# ## for test get iou
# proposals_idx =   torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 							  [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3]]).int().cuda()
#
# instance_labels = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
# 								[0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3]]).int().cuda()
#
# proposals_num = 4
# instance_num = torch.tensor([3, 4]).int().cuda()
#
# res = pointnet2_utils.get_iou(proposals_idx, proposals_num, instance_labels, instance_num)
#
# print(res)
