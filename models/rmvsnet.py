import torch
import torch.nn as nn
import numpy as np
from os import path
import torch.nn.functional as F

from .gru import GRU
from .unet_ds2gn import UNetDS2GN

from .warping import get_homographies, warp_homographies
import pdb

class RMVSNet(nn.Module):
    def __init__(self, train=False): # 参数并没有使用到
        super(RMVSNet, self).__init__()
        # setup network modules
        # 特征提取模块
        self.feature_extractor = UNetDS2GN()

        gru_input_size = self.feature_extractor.output_size
        gru1_output_size = 16
        gru2_output_size = 4
        gru3_output_size = 2
        # GRU模块
        self.gru1 = GRU(gru_input_size, gru1_output_size, 3)
        self.gru2 = GRU(gru1_output_size, gru2_output_size, 3)
        self.gru3 = GRU(gru2_output_size, gru3_output_size, 3)

        self.prob_conv = nn.Conv2d(2, 1, 3, 1, 1)

        file_path = path.dirname(path.abspath(__file__))
        pretrained_weights_file = path.join(file_path,
                                            'RMVSNet-pretrained.pth')
        pretrained_weights = torch.load(pretrained_weights_file)
        self.load_state_dict(pretrained_weights)

    def compute_cost_volume(self, warped):
        '''
        Warped: N x C x M x H x W

        returns: 1 x C x M x H x W
        '''
        warped_sq = warped ** 2
        av_warped = warped.mean(0)
        av_warped_sq = warped_sq.mean(0)
        cost = av_warped_sq - (av_warped ** 2)

        return cost.unsqueeze(0)

    def compute_depth(self, prob_volume, depth_start, depth_interval, depth_num):
        '''
        prob_volume: 1 x D x H x W
        '''
        _, M, H, W = prob_volume.shape
        # prob_indices = HW shaped vector
        probs, indices = prob_volume.max(1)
        depth_range = depth_start + torch.arange(depth_num).float() * depth_interval
        depth_range = depth_range.to(prob_volume.device)
        depths = torch.index_select(depth_range, 0, indices.flatten())
        depth_image = depths.view(H, W)
        prob_image = probs.view(H, W)

        return depth_image, prob_image



    def forward(self, images, intrinsics, extrinsics, depth_start, depth_interval, depth_num):
        '''
        Takes all entry and outputs probability volume

        N x D x H x W probability map
        '''
        N, C, IH, IW = images.shape
        f = self.feature_extractor(images)

        Hs = get_homographies(f, intrinsics, extrinsics, depth_start, depth_interval, depth_num)

        # N, C, D, H, W = warped.shape
        cost_1 = None
        cost_2 = None
        cost_3 = None
        depth_costs = []

        # ref_f = f[0]
        # ref_f2 = ref_f ** 2
        # 使用一个循环利用上一层的数据
        for d in range(depth_num):
            # mean_f = ref_f
            # mean_f2 = ref_f2
            # warped = N x C x H x W
            ref_f = f[:1]
            warped = warp_homographies(f[1:], Hs[1:, d])
            all_f = torch.cat((ref_f, warped), 0)

            # cost_d = 1 x C x H x W
            cost_d =  self.compute_cost_volume(all_f)
            cost_1 = self.gru1(-cost_d, cost_1)
            cost_2 = self.gru2(cost_1, cost_2)
            cost_3 = self.gru3(cost_2, cost_3)

            reg_cost = self.prob_conv(cost_3)
            depth_costs.append(reg_cost)

        prob_volume = torch.cat(depth_costs, 1)
        softmax_probs = torch.softmax(prob_volume, 1)


        # compute depth map from prob / depth values
        return self.compute_depth(softmax_probs, depth_start, depth_interval, depth_num)

# 定义分类损失函数
def r_mvsnet_loss(prob_volume, gt_depth_image, depth_num, depth_start, depth_interval,mask):
    #get depth mask
    # mask = mask > 0.5
    mask_true = torch.ne(gt_depth_image, 0.0).to('float32')
    valid_pixel_num = torch.sum(mask_true, [1,2,3]) + 1e-7
    # gt depth map -> gt index map
    shape = gt_depth_image.shape
    depth_end = depth_start + (depth_num.to('float32') -1) * depth_interval
    start_mat = depth_start.view(shape[0], 1, 1, 1).repeat(1, shape[1], shape[2], 1)
    interval_mat = depth_interval.view(shape[0], 1, 1, 1).repeat(1, shape[1], shape[2], 1)

    gt_index_image = torch.div(gt_depth_image - start_mat, interval_mat)
    gt_index_image = torch.multiply(mask_true, gt_index_image)
    gt_index_image = torch.round(gt_index_image).to('int32')
    # gt index map -> gt one hot volume (B x H x W x 1)
    gt_index_volume = F.one_hot(gt_index_image, depth_num, axis=1)
    # cross entropy image (B x H x W x 1)
    cross_entropy_image = -(gt_index_volume * torch.log(prob_volume)).sum(1)

    # masked cross entropy loss
    masked_cross_entropy_image = torch.multiply(mask_true, cross_entropy_image)
    masked_cross_entropy = tf.reduce_sum(masked_cross_entropy_image, axis=[1, 2, 3])
    masked_cross_entropy = tf.reduce_sum(masked_cross_entropy / valid_pixel_num)
    # return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
