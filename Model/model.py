import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import spconv
from spconv.modules import SparseModule
import functools
from collections import OrderedDict

from lib.pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils
from lib.pointgroup_ops.functions import pointgroup_ops

from util import utils

################################################################################################
cnt = 0

def save4visualize(batch, ret):
    dst = "/media/xue/DATA1/xue/temp/vis_output_epoch368"

    xyz = batch['locs_float'].numpy()  # np.float, (N, 3)
    labels = batch['labels'].numpy()  # np.int, (N)
    instance_labels = batch['instance_labels'].numpy()
    batch_offsets = batch['offsets'].numpy()  # (B + 1, )

    semantic_scores = ret['semantic_scores'].cpu().detach().numpy()  # (N, nClasses)
    embedding_feats = ret['embedding_feats'].cpu().detach().numpy()  # (N, nDim)
    pos_feats = ret['pos_feats'].cpu().detach().numpy()  # (N, nDim)
    pt_offsets = ret['pt_offsets'].cpu().detach().numpy()  # (N, 3)

    proposals_idx = ret['proposals_idx'].cpu().detach().numpy()  # (N', 2) 0 for proposal_label, 1 for point_idx in N
    proposals_offset = ret['proposals_offset'].cpu().detach().numpy()  # (nProposal + 1, )
    object_idxs = ret['object_idxs'].cpu().detach().numpy()  # (N', )
    candidate_idx = ret['candidate_idx'].cpu().detach().numpy()  # (M, ) index from object points

    colors = np.array([[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                       for _ in range(200)]).astype(np.int)

    # candidate
    candidate_xyz = xyz[object_idxs][candidate_idx]
    candidate_color = np.array([[0, 0, 255] for _ in range(candidate_xyz.shape[0])])

    candidate_offset = pt_offsets[object_idxs][candidate_idx]
    candidate_xyz_offset = candidate_xyz + candidate_offset
    candidate_offset_color = np.array([[255, 0, 0] for _ in range(candidate_xyz.shape[0])])

    # pred
    sem_pred = np.argmax(semantic_scores, axis=-1)
    sem_pred_color = colors[sem_pred]

    ins_pred_color = np.array([[0, 0, 0] for _ in range(xyz.shape[0])])
    ins_pred_color[proposals_idx[:, 1]] = colors[proposals_idx[:, 0]]

    # gt
    sem_gt_color = colors[labels]
    ins_gt_color = colors[instance_labels]

    global cnt

    with open(os.path.join(dst, str(cnt) + "_candidate_xyz.txt"), "w") as writer:
        for i in range(candidate_xyz.shape[0]):
            writer.write(str(candidate_xyz[i][0]) + " " + str(candidate_xyz[i][1]) + " " + str(candidate_xyz[i][2]) + " " +
                         str(candidate_color[i][0]) + " " + str(candidate_color[i][1]) + " " + str(candidate_color[i][2]) +
                         "\n")
    with open(os.path.join(dst, str(cnt) + "_candidate_xyz_offset.txt"), "w") as writer:
        for i in range(candidate_xyz_offset.shape[0]):
            writer.write(str(candidate_xyz_offset[i][0]) + " " + str(candidate_xyz_offset[i][1]) + " " + str(candidate_xyz_offset[i][2]) + " " +
                         str(candidate_offset_color[i][0]) + " " + str(candidate_offset_color[i][1]) + " " + str(candidate_offset_color[i][2]) +
                         "\n")

    # with open(os.path.join(dst, str(cnt) + "_sem_pred.txt"), "w") as writer:
    #     for i in range(xyz.shape[0]):
    #         writer.write(str(xyz[i][0]) + " " + str(xyz[i][1]) + " " + str(xyz[i][2]) + " " +
    #                      str(sem_pred_color[i][0]) + " " + str(sem_pred_color[i][1]) + " " + str(sem_pred_color[i][2]) +
    #                      "\n")

    with open(os.path.join(dst, str(cnt) + "_ins_merge_pred.txt"), "w") as writer:
        for i in range(xyz.shape[0]):
            writer.write(str(xyz[i][0]) + " " + str(xyz[i][1]) + " " + str(xyz[i][2]) + " " +
                         str(ins_pred_color[i][0]) + " " + str(ins_pred_color[i][1]) + " " + str(ins_pred_color[i][2]) +
                         "\n")

    # with open(os.path.join(dst, str(cnt) + "_sem_gt.txt"), "w") as writer:
    #     for i in range(xyz.shape[0]):
    #         writer.write(str(xyz[i][0]) + " " + str(xyz[i][1]) + " " + str(xyz[i][2]) + " " +
    #                      str(sem_gt_color[i][0]) + " " + str(sem_gt_color[i][1]) + " " + str(sem_gt_color[i][2]) +
    #                      "\n")
    #
    # with open(os.path.join(dst, str(cnt) + "_ins_gt.txt"), "w") as writer:
    #     for i in range(xyz.shape[0]):
    #         writer.write(str(xyz[i][0]) + " " + str(xyz[i][1]) + " " + str(xyz[i][2]) + " " +
    #                      str(ins_gt_color[i][0]) + " " + str(ins_gt_color[i][1]) + " " + str(ins_gt_color[i][2]) +
    #                      "\n")

    cnt += 1

################################################################################################


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
                  for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False,
                                    indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id + 1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False,
                                           indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn,
                                                         indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat((identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)

        return output


class InstanceSegPipline(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_c = cfg.input_channel
        m = cfg.m
        classes = cfg.classes
        block_reps = cfg.block_reps
        block_residual = cfg.block_residual

        self.cfg = cfg

        self.score_scale = cfg.score_scale
        self.score_fullscale = cfg.score_fullscale
        self.mode = cfg.score_mode

        self.prepare_epochs = cfg.prepare_epochs

        self.pretrain_path = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module
        self.fix_module = cfg.fix_module

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if cfg.use_coords:
            input_c += 3

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )

        self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                           indice_key_id=1)

        self.output_layer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )

        #### semantic segmentation
        self.semantic_encoder = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )
        self.semantic_linear = nn.Linear(m, classes)  # bias(default): True

        #### instance-aware embedding feature extractor
        self.embedding_encoder = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )

        #### position encoder
        self.position_encoder = nn.Sequential(
            nn.Linear(3, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )

        #### offset
        self.offset_encoder = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(m, 3, bias=True)

        #### score branch
        self.score_encoder = nn.Sequential(
            nn.Linear(3 * m, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )
        self.score_unet = UBlock([m, 2 * m], norm_fn, 2, block, indice_key_id=1)
        self.score_outputlayer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )
        self.score_linear = nn.Linear(m, 1)

        self.apply(self.set_bn_init)

        #### fix parameter
        module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
                      'semantic_encoder': self.semantic_encoder, 'semantic_linear': self.semantic_linear,
                      'embedding_encoder': self.embedding_encoder, 'position_encoder': self.position_encoder,
                      'offset_encoder': self.offset_encoder, 'offset_linear': self.offset_linear,
                      'score_unet': self.score_unet, 'score_outputlayer': self.score_outputlayer,
                      'score_linear': self.score_linear}

        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

        #### load pretrain weights
        if self.pretrain_path is not None:
            pretrain_dict = torch.load(self.pretrain_path)
            for m in self.pretrain_module:
                print(
                    "Load pretrained " + m + ": %d/%d" % utils.load_model_param(module_map[m], pretrain_dict, prefix=m))

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    @staticmethod
    def cal_connection_components(adjacency):
        def bfs(node):
            cc = [node]
            q = [node]
            visit[node] = 1
            while q:
                node = q.pop()
                for neihbor in nodes:
                    if not visit[neihbor] and adjacency[node][neihbor] == 1:
                        q.append(neihbor)
                        cc.append(neihbor)
                        visit[neihbor] = 1
            nonlocal ccs
            ccs.append(cc)
        n = adjacency.size(0)
        nodes = range(n)
        visit = [0] * n
        ccs = []
        for i in nodes:
            if not visit[i]:
                bfs(i)
        return ccs

    def proposal_generation(self, feats, candidate_feats, candidate_xyz, batch_offsets, candidates_batch_offsets):
        proposal_idx = []
        proposals_offset = []
        proposal_num_batch = [0]

        for i in range(batch_offsets.size(0) - 1):
            feats_batch_i = feats[batch_offsets[i]: batch_offsets[i + 1]]  # (N_i, F)
            candidate_feats_batch_i = candidate_feats[
                                      candidates_batch_offsets[i]: candidates_batch_offsets[i + 1]]  # (K_i, F)
            candidate_xyz_batch_i = candidate_xyz[candidates_batch_offsets[i]: candidates_batch_offsets[i + 1]]
            ncandidate = candidate_feats_batch_i.size(0)

            ## relation matrix between all points and candidate points [N_i, K_i]
            relation_matrix = torch.norm(feats_batch_i.unsqueeze(1) - candidate_feats_batch_i.unsqueeze(0), dim=-1, p=2)
            proposal_idx_i = torch.argmin(relation_matrix, dim=-1)  # (N_i)
            del relation_matrix  # too large, release memory as long as it is done

            ## merge the proposal
            # relation matrix between candidate points for merging [K_i, K_i]
            relation_matrix_proposals = torch.norm(candidate_xyz_batch_i.unsqueeze(1) -
                                                   candidate_xyz_batch_i.unsqueeze(0), dim=-1, p=2)
            adjacency = torch.zeros_like(relation_matrix_proposals, dtype=torch.long)
            adjacency[relation_matrix_proposals < 8] = 1  # 0-1 matrix, 1 for two proposals belong to each other
            # calculate the connection components of the relation matrix
            ccs = self.cal_connection_components(adjacency)
            del relation_matrix_proposals, adjacency

            # remap the proposal label to the merged label, from 0 to nproposal
            remapper = torch.arange(0, ncandidate, device="cuda").long()
            for cc_i in range(len(ccs)):
                remapper[ccs[cc_i]] = cc_i
            remapper = utils.get_merged_proposal_labels(remapper)
            proposal_idx_i = remapper[proposal_idx_i]
            proposal_idx_i, proposal_point_idx_i = torch.sort(proposal_idx_i)  # 因为pointgroup的接口需要把proposal聚集一块处理
            proposal_num = torch.unique(proposal_idx_i).size(0)
            proposal_num_batch.append(proposal_num)

            proposals_offset_i = utils.get_batch_offsets(proposal_idx_i, proposal_num)  # proposal offset (nProposal + 1)
            proposals_offset.append(proposals_offset_i)

            proposal_idx_i, proposal_point_idx_i = proposal_idx_i.unsqueeze(1), proposal_point_idx_i.unsqueeze(1)
            proposal_idx_i = torch.cat((proposal_idx_i, proposal_point_idx_i), dim=1).int()
            proposal_idx.append(proposal_idx_i)

        ## proposal_idx and proposal_offset batch correct
        for i in range(1, len(proposals_offset)):
            proposals_offset[i] = proposals_offset[i] + proposals_offset[i - 1][-1]
            proposals_offset[i] = proposals_offset[i][1:]

            # (put them as [proposal_idx_bactch1, proposal_idx_bactch2 + nproposal_bactch1, ....])
            proposal_idx[i][0] += proposal_num_batch[i]

        ## concat proposal idx and proposal offset
        proposal_idx = torch.cat(proposal_idx, dim=0).contiguous()
        proposals_offset = torch.cat(proposals_offset, dim=0).contiguous()

        return proposal_idx, proposals_offset

    @staticmethod
    def select(xyz, feats, xyz_batch_cnt, candidates_batch_cnt):
        """
        FPS and then select top k candidate by considering feature entropy
        :param xyz: input xyz of each points [N, 3]
        :param feats: input feature of each points [N, F]
        :return: index of instance candidate points
        """

        def cal_entropy(feats):
            """
            :param feats: [N, F]
            :return:
            """
            softmax_feat = F.softmax(feats, dim=-1)
            feat_entropy = torch.sum(-softmax_feat * torch.log(softmax_feat), dim=-1)  # sigma{-fi * log{fi}}, [B, N]
            return feat_entropy

        # assert xyz.size(1) == feats.size(1)
        # points_num = xyz.size(1)

        candidate_idx = pointnet2_utils.stack_farthest_point_sample(xyz, xyz_batch_cnt,
                                                                    candidates_batch_cnt).long()  # fps采样，返回index [B, NSample]
        # candidate_idx = pointnet2_utils.furthest_point_sample(xyz, candidates_num)  # fps采样，返回index [B, NSample]
        # idx = pointnet2_utils.furthest_point_sample(xyz, points_num // 2)  # fps采样，返回index [B, NSample]
        # gather_feature = pointnet2_utils.gather_operation(feats.permute(0, 2, 1).contiguous(),
        #                                                   idx).permute(0, 2, 1).contiguous()  # 根据index获取采样点
        # gather_feature = gather_feature.cpu()
        #
        # feat_entropy = cal_entropy(gather_feature)
        # sorted_idx = feat_entropy.sort(dim=-1)[1][:, :candidates_num].long()  # [B, NSample]
        # sorted_idx = sorted_idx.to("cuda")
        # candidate_idx = torch.gather(idx, -1, sorted_idx)

        return candidate_idx

    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode):
        '''
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        '''
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean = pointgroup_ops.sec_mean(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_mean = torch.index_select(clusters_coords_mean, 0,
                                                  clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = pointgroup_ops.sec_min(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_max = pointgroup_ops.sec_max(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float

        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[
            0] - 0.01  # (nCluster), float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = - min_xyz + torch.clamp(fullscale - range - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(
            fullscale - range + 0.001, max=0) * torch.rand(3).cuda()
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()],
                                    1)  # (sumNPoint, 1 + 3)

        out_coords, inp_map, out_map = pointgroup_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1,
                                                                       mode)
        # output_coords: M * (1 + 3) long
        # input_map: sumNPoint int
        # output_map: M * (maxActive + 1) int

        out_feats = pointgroup_ops.voxelization(clusters_feats, out_map.cuda(), mode)  # (M, C), float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape,
                                                     int(clusters_idx[-1, 0]) + 1)

        return voxelization_feats, inp_map

    def forward(self, input, input_map, coords, batch_idxs, batch_offsets, epoch):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        '''
        ret = {}

        # backbone
        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]

        # semantice segmentation
        semantic_feats = self.semantic_encoder(output_feats)
        semantic_scores = self.semantic_linear(semantic_feats)
        semantic_preds = semantic_scores.max(dim=-1)[1]
        ret['semantic_scores'] = semantic_scores

        # instance-aware embedding feature encoding
        embedding_feats = self.embedding_encoder(output_feats)
        ret['embedding_feats'] = embedding_feats

        # position encoding
        pos_feats = self.position_encoder(coords)
        ret['pos_feats'] = pos_feats

        # offset encoding
        pt_offsets_feats = self.offset_encoder(output_feats)
        pt_offsets = self.offset_linear(pt_offsets_feats)  # (N, 3), float32
        ret['pt_offsets'] = pt_offsets

        feats = torch.cat((semantic_feats, embedding_feats, pos_feats), dim=-1)

        if (epoch >= self.prepare_epochs):
            #### get prooposal clusters
            object_idxs = torch.nonzero(semantic_preds > 0).view(-1)  # 0 for non-object class

            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input.batch_size)
            coords_ = coords[object_idxs]
            feats_ = torch.cat([embedding_feats[object_idxs], pos_feats[object_idxs]], dim=-1)
            # feats_ = feats[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]

            xyz_batch_cnt = torch.tensor([batch_offsets_[i] - batch_offsets_[i - 1]
                                          for i in range(1, batch_offsets_.size(0))],
                                         device=batch_offsets_.device).int()  # (B,)

            # assume the num_candidates for each batch are different
            candidates_batch_cnt = xyz_batch_cnt // self.cfg.candidate_scale
            candidates_batch_cnt = torch.clamp(candidates_batch_cnt, 10, self.cfg.max_candidate).int()

            candidates_batch_offsets = torch.tensor([0] * batch_offsets_.size(0),
                                                    device=batch_offsets_.device).int()  # (B + 1,)
            for i in range(candidates_batch_cnt.size(0)):
                candidates_batch_offsets[i + 1] = candidates_batch_offsets[i] + candidates_batch_cnt[i]

            candidate_idx = self.select(coords_, feats_, xyz_batch_cnt, candidates_batch_cnt)  # select module
            candidate_feats = feats_[candidate_idx]  # (M, F)
            candidate_offset = pt_offsets_[candidate_idx]
            candidate_xyz_offseted = coords_[candidate_idx] + candidate_offset
            # proposal generation
            proposals_idx, proposals_offset = self.proposal_generation(feats_, candidate_feats, candidate_xyz_offseted,
                                                                       batch_offsets_, candidates_batch_offsets)
            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
            # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int

            torch.cuda.empty_cache()

            # # for debug to output some intermediate results
            # ret['proposals_idx'] = proposals_idx
            # ret['proposals_offset'] = proposals_offset
            # ret['object_idxs'] = object_idxs
            # ret['candidate_idx'] = candidate_idx

            #### proposals voxelization again
            score_feats = self.score_encoder(feats)
            input_feats, inp_map = self.clusters_voxelization(proposals_idx.cpu(), proposals_offset.cpu(), score_feats,
                                                              coords, self.score_fullscale, self.score_scale, self.mode)

            #### score
            score = self.score_unet(input_feats)
            score = self.score_outputlayer(score)
            score_feats = score.features[inp_map.long()]  # (sumNPoint, C)
            score_feats = pointgroup_ops.roipool(score_feats, proposals_offset.cuda())  # (nProposal, C)
            scores = self.score_linear(score_feats)  # (nProposal, 1)

            ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

        return ret


def model_fn_decorator(test=False):
    from util.config import cfg

    semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).to(cfg.device)
    score_criterion = nn.BCELoss(reduction='none').to(cfg.device)

    def test_model_fn(batch, model, epoch):
        pass

    def train_model_fn(batch, model, epoch):
        ##### prepare input and forward
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
        # 'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
        # 'id': tbl, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
        coords = batch['locs'].cuda()  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()  # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()  # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        feats = batch['feats'].cuda()  # (N, C), float32, cuda
        labels = batch['labels'].cuda()  # (N), long, cuda
        instance_labels = batch['instance_labels'].cuda()  # (N), long, cuda, 0~total_nInst, -100

        instance_info = batch['instance_info'].cuda()  # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda

        batch_offsets = batch['offsets'].cuda()  # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        # get and unpack model result
        ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch)

        # # for debug to output some intermediate results
        # save4visualize(batch, ret)

        embedding_feats = ret['embedding_feats']  # [B, N, F] embedding feats for calculating the embedding loss
        semantic_scores = ret['semantic_scores']  # [N, nclass]
        pt_offsets = ret['pt_offsets']  # (N, 3), float32, cuda
        if (epoch > cfg.prepare_epochs):
            scores, proposals_idx, proposals_offset = ret['proposal_scores']
            # scores: (nProposal, 1) float, cuda
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu

        loss_inp = {}
        loss_inp['semantic_scores'] = (semantic_scores, labels)
        loss_inp['embedding_feats'] = embedding_feats
        loss_inp['pt_offsets'] = (pt_offsets, coords_float, instance_info, instance_labels)
        if (epoch > cfg.prepare_epochs):
            loss_inp['proposal_scores'] = (scores, proposals_idx, proposals_offset, instance_pointnum)

        loss, loss_out, infos = loss_fn(loss_inp, epoch)

        ##### accuracy / visual_dict / meter_dict
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores
            preds['pt_offsets'] = pt_offsets
            if (epoch > cfg.prepare_epochs):
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)

            visual_dict = {}
            visual_dict['loss'] = loss
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}
            meter_dict['loss'] = (loss.item(), coords.shape[0])
            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])

        return loss, preds, visual_dict, meter_dict

    def loss_fn(loss_inp, epoch):

        loss_out = {}
        infos = {}

        '''semantic loss'''
        semantic_scores, semantic_labels = loss_inp['semantic_scores']
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_loss = semantic_criterion(semantic_scores, semantic_labels)
        loss_out['semantic_loss'] = (semantic_loss, semantic_scores.shape[0])

        '''offset loss'''
        pt_offsets, coords, instance_info, instance_labels = loss_inp['pt_offsets']
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        # instance_labels: (N), long

        gt_offsets = instance_info[:, 0:3] - coords  # (N, 3)
        pt_diff = pt_offsets - gt_offsets  # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # (N)
        valid = (instance_labels != cfg.ignore_label).float()
        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)  # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

        loss_out['offset_norm_loss'] = (offset_norm_loss, valid.sum())
        loss_out['offset_dir_loss'] = (offset_dir_loss, valid.sum())

        '''discriminative loss'''
        embedding_feats = loss_inp['embedding_feats']
        # embedding_feats: (B, N, F)

        instance_labels_unique = torch.unique(instance_labels)
        instance_labels_unique = instance_labels_unique[instance_labels_unique != cfg.ignore_label]
        instance_num = instance_labels_unique.size(0)

        # pull loss
        pull_loss = 0.0
        mu_feats = []  # the mean feature for a specific instance
        for i in instance_labels_unique:
            instance_feats_i = embedding_feats[instance_labels == i]  # [instance_i_pnum, F]

            mu_feat_i = torch.mean(instance_feats_i, dim=0)  # [F,]
            mu_feats.append(mu_feat_i)

            pull_loss_i = torch.norm(instance_feats_i - mu_feat_i, dim=-1, p=2) - cfg.sigma_1
            pull_loss_i = torch.where(pull_loss_i >= 0, pull_loss_i,
                                      torch.zeros_like(pull_loss_i)) ** 2  # max(0, ||mu - fi||2 - sigma1)^2
            pull_loss += torch.mean(pull_loss_i)
        pull_loss /= instance_num

        loss_out['pull_loss'] = (pull_loss, instance_num)

        # push loss
        push_loss = 0.0
        for i, feat_i in enumerate(instance_labels_unique):
            for j, feat_j in enumerate(instance_labels_unique):
                if i != j:
                    mu_feat_i = mu_feats[i]
                    mu_feat_j = mu_feats[j]

                    push_loss_ij = max(cfg.sigma_2 - torch.norm(mu_feat_i - mu_feat_j, dim=-1, p=2), 0) ** 2
                    push_loss += push_loss_ij
        push_loss /= (instance_num * (instance_num - 1))

        loss_out['push_loss'] = (push_loss, instance_num * (instance_num - 1))

        # reg loss
        reg_loss = 0.0
        for mu_feat in mu_feats:
            reg_loss += torch.norm(mu_feat, dim=-1, p=2)
        reg_loss /= instance_num

        loss_out['reg_loss'] = (reg_loss, instance_num)

        embedding_loss = cfg.weight_pull * pull_loss + cfg.weight_push * push_loss + cfg.weight_reg * reg_loss

        loss_out['embedding_loss'] = (embedding_loss, instance_num)

        if (epoch > cfg.prepare_epochs):
            '''score loss'''
            scores, proposals_idx, proposals_offset, instance_pointnum = loss_inp['proposal_scores']
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int

            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].contiguous(), proposals_offset, instance_labels,
                                          instance_pointnum)  # (nProposal, nInstance), float
            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
            gt_scores = get_segmented_scores(gt_ious, cfg.fg_thresh, cfg.bg_thresh)

            score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            score_loss = score_loss.mean()

            loss_out['score_loss'] = (score_loss, gt_ious.shape[0])

        ## total loss
        loss = semantic_loss + embedding_loss + offset_norm_loss + offset_dir_loss

        if epoch > cfg.prepare_epochs:
            loss += score_loss

        return loss, loss_out, infos

    def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
        '''
        :param scores: (N), float, 0~1
        :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
        '''
        fg_mask = scores > fg_thresh
        bg_mask = scores < bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)

        segmented_scores = (fg_mask > 0).float()
        k = 1 / (fg_thresh - bg_thresh)
        b = bg_thresh / (bg_thresh - fg_thresh)
        segmented_scores[interval_mask] = scores[interval_mask] * k + b

        return segmented_scores

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
