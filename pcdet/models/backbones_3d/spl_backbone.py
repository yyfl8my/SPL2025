from functools import partial

import math
import numpy as np
import torch
import torch.nn as nn
import torch_scatter

import sys
sys.path.append('/public/home/wangcb/LION-main/pcdet/ops/mamba')

from mamba_ssm import Block as MambaBlock

from torch.nn import functional as F

from ..model_utils.retnet_attn import Block as RetNetBlock
from ..model_utils.rwkv_cls import Block as RWKVBlock
from ..model_utils.vision_lstm2 import xLSTM_Block
from ..model_utils.ttt import TTTBlock
from ...utils.spconv_utils import replace_feature, spconv
import torch.utils.checkpoint as cp


@torch.inference_mode()
def get_window_coors_shift_v2(coords, sparse_shape, window_shape, shift=False):
    sparse_shape_z, sparse_shape_y, sparse_shape_x = sparse_shape
    win_shape_x, win_shape_y, win_shape_z = window_shape

    if shift:
        shift_x, shift_y, shift_z = win_shape_x // 2, win_shape_y // 2, win_shape_z // 2
    else:
        shift_x, shift_y, shift_z = 0, 0, 0  # win_shape_x, win_shape_y, win_shape_z

    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1)  # plus one here to meet the needs of shift.

    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    x = coords[:, 3] + shift_x
    y = coords[:, 2] + shift_y
    z = coords[:, 1] + shift_z

    win_coors_x = x // win_shape_x
    win_coors_y = y // win_shape_y
    win_coors_z = z // win_shape_z

    coors_in_win_x = x % win_shape_x
    coors_in_win_y = y % win_shape_y
    coors_in_win_z = z % win_shape_z

    batch_win_inds_x = coords[:, 0] * max_num_win_per_sample + win_coors_x * max_num_win_y * max_num_win_z + \
                       win_coors_y * max_num_win_z + win_coors_z
    batch_win_inds_y = coords[:, 0] * max_num_win_per_sample + win_coors_y * max_num_win_x * max_num_win_z + \
                       win_coors_x * max_num_win_z + win_coors_z

    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)

    return batch_win_inds_x, batch_win_inds_y, coors_in_win


def get_window_coors_shift_v1(coords, sparse_shape, window_shape):
    _, m, n = sparse_shape
    n2, m2, _ = window_shape

    n1 = int(np.ceil(n / n2) + 1)  # plus one here to meet the needs of shift.
    m1 = int(np.ceil(m / m2) + 1)  # plus one here to meet the needs of shift.

    x = coords[:, 3]
    y = coords[:, 2]

    x1 = x // n2
    y1 = y // m2
    x2 = x % n2
    y2 = y % m2

    return 2 * n2, 2 * m2, 2 * n1, 2 * m1, x1, y1, x2, y2

class FlattenedWindowMapping(nn.Module):
    def __init__(
            self,
            window_shape,
            group_size,
            shift,
            win_version='v2'
    ) -> None:
        super().__init__()
        self.window_shape = window_shape
        self.group_size = group_size
        self.win_version = win_version
        self.shift = shift

    def forward(self, coords: torch.Tensor, batch_size: int, sparse_shape: list):
        coords = coords.long()
        _, num_per_batch = torch.unique(coords[:, 0], sorted=False, return_counts=True)
        batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))
        num_per_batch_p = (
                torch.div(
                    batch_start_indices[1:] - batch_start_indices[:-1] + self.group_size - 1,
                    self.group_size,
                    rounding_mode="trunc",
                )
                * self.group_size
        )

        batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0), (1, 0))
        flat2win = torch.arange(batch_start_indices_p[-1], device=coords.device)  # .to(coords.device)
        win2flat = torch.arange(batch_start_indices[-1], device=coords.device)  # .to(coords.device)

        for i in range(batch_size):
            if num_per_batch[i] != num_per_batch_p[i]:
                
                bias_index = batch_start_indices_p[i] - batch_start_indices[i]
                flat2win[
                    batch_start_indices_p[i + 1] - self.group_size + (num_per_batch[i] % self.group_size):
                    batch_start_indices_p[i + 1]
                    ] = flat2win[
                        batch_start_indices_p[i + 1]
                        - 2 * self.group_size
                        + (num_per_batch[i] % self.group_size): batch_start_indices_p[i + 1] - self.group_size
                        ] if (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) - self.group_size != 0 else \
                        win2flat[batch_start_indices[i]: batch_start_indices[i + 1]].repeat(
                            (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) // num_per_batch[i] + 1)[
                        : self.group_size - (num_per_batch[i] % self.group_size)] + bias_index


            win2flat[batch_start_indices[i]: batch_start_indices[i + 1]] += (
                    batch_start_indices_p[i] - batch_start_indices[i]
            )

            flat2win[batch_start_indices_p[i]: batch_start_indices_p[i + 1]] -= (
                    batch_start_indices_p[i] - batch_start_indices[i]
            )

        mappings = {"flat2win": flat2win, "win2flat": win2flat}

        get_win = self.win_version

        if get_win == 'v1':
            for shifted in [False]:
                (
                    n2,
                    m2,
                    n1,
                    m1,
                    x1,
                    y1,
                    x2,
                    y2,
                ) = get_window_coors_shift_v1(coords, sparse_shape, self.window_shape)
                vx = (n1 * y1 + (-1) ** y1 * x1) * n2 * m2 + (-1) ** y1 * (m2 * x2 + (-1) ** x2 * y2)
                vx += coords[:, 0] * sparse_shape[2] * sparse_shape[1] * sparse_shape[0]
                vy = (m1 * x1 + (-1) ** x1 * y1) * m2 * n2 + (-1) ** x1 * (n2 * y2 + (-1) ** y2 * x2)
                vy += coords[:, 0] * sparse_shape[2] * sparse_shape[1] * sparse_shape[0]
                _, mappings["x" + ("_shift" if shifted else "")] = torch.sort(vx)
                _, mappings["y" + ("_shift" if shifted else "")] = torch.sort(vy)

        elif get_win == 'v2':
            batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2(coords, sparse_shape,
                                                                                         self.window_shape, self.shift)
            vx = batch_win_inds_x * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vx += coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                  self.window_shape[2] + coors_in_win[..., 0]

            vy = batch_win_inds_y * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vy += coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                  self.window_shape[2] + coors_in_win[..., 0]

            _, mappings["x"] = torch.sort(vx)
            _, mappings["y"] = torch.sort(vy)

        elif get_win == 'v3':
            batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2(coords, sparse_shape,
                                                                                         self.window_shape)
            vx = batch_win_inds_x * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vx_xy = vx + coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                    self.window_shape[2] + coors_in_win[..., 0]
            vx_yx = vx + coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                    self.window_shape[2] + coors_in_win[..., 0]

            vy = batch_win_inds_y * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vy_xy = vy + coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                    self.window_shape[2] + coors_in_win[..., 0]
            vy_yx = vy + coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                    self.window_shape[2] + coors_in_win[..., 0]

            _, mappings["x_xy"] = torch.sort(vx_xy)
            _, mappings["y_xy"] = torch.sort(vy_xy)
            _, mappings["x_yx"] = torch.sort(vx_yx)
            _, mappings["y_yx"] = torch.sort(vy_yx)

        return mappings


class PatchMerging3D(nn.Module):
    def __init__(self, dim, out_dim=-1, down_scale=[2, 2, 2], norm_layer=nn.LayerNorm, diffusion=False, diff_scale=0.2):
        super().__init__()
        self.dim = dim

        self.sub_conv = spconv.SparseSequential(
            spconv.SubMConv3d(dim, dim, 3, bias=False, indice_key='subm'),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        if out_dim == -1:
            self.norm = norm_layer(dim)
        else:
            self.norm = norm_layer(out_dim)

        self.sigmoid = nn.Sigmoid()
        self.down_scale = down_scale
        self.diffusion = diffusion
        self.diff_scale = diff_scale
        # 新增三向注意力模块
        self.triple_att = TripleAtt(feat_dim=dim)

        self.num_points = 6  # 3


    def forward(self, x, coords_shift=1, diffusion_scale=4):
        assert diffusion_scale == 4 or diffusion_scale == 2
        x = self.sub_conv(x)

        d, h, w = x.spatial_shape
        down_scale = self.down_scale

        # 辅助函数：计算每个体素的局部响应（在26邻域内求均值）
        def compute_local_response(indices, responses, d, h, w):
            # indices: (N, 4), responses: (N,)
            device = indices.device
            b = indices[:, 0].to(torch.long)
            z = indices[:, 1].to(torch.long)
            y = indices[:, 2].to(torch.long)
            x_coord = indices[:, 3].to(torch.long)
            keys = b * (d * h * w) + z * (h * w) + y * w + x_coord
            sorted_keys, sort_idx = keys.sort()
            sorted_responses = responses[sort_idx]
            local_sum = torch.zeros_like(responses)
            local_count = torch.zeros_like(responses)
            offsets = []
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        offsets.append((dz, dy, dx))
            for dz, dy, dx in offsets:
                z_n = z + dz
                y_n = y + dy
                x_n = x_coord + dx
                valid = (z_n >= 0) & (z_n < d) & (y_n >= 0) & (y_n < h) & (x_n >= 0) & (x_n < w)
                if valid.sum() == 0:
                    continue
                candidate_keys = b[valid] * (d * h * w) + z_n[valid] * (h * w) + y_n[valid] * w + x_n[valid]
                pos = torch.searchsorted(sorted_keys, candidate_keys)
                valid_pos = pos < sorted_keys.size(0)
                pos = pos[valid_pos]
                candidate_keys_valid = candidate_keys[valid_pos]
                mask_match = sorted_keys[pos] == candidate_keys_valid
                original_idx = torch.nonzero(valid).squeeze(1)[valid_pos][mask_match]
                local_sum[original_idx] += sorted_responses[pos[mask_match]]
                local_count[original_idx] += 1
            local_avg = local_sum / (local_count + 1e-6)
            return local_avg       

        if self.diffusion:
            # 计算均值
            own_response = x.features.mean(-1)
            # 归一化均值
            own_response_min, own_response_max = own_response.min(), own_response.max()
            own_response = (own_response - own_response_min) / (own_response_max - own_response_min + 1e-6)

            # 计算局部相应
            local_resp = compute_local_response(x.indices, own_response, d, h, w)
            # 归一化局部响应
            local_resp_min, local_resp_max = local_resp.min(), local_resp.max()
            local_resp = (local_resp - local_resp_min) / (local_resp_max - local_resp_min + 1e-6)

            # 计算最大响应
            max_response = x.features.max(dim=-1)[0]
             # 归一化最大响应
            max_response_min, max_response_max = max_response.min(), max_response.max()
            max_response = (max_response - max_response_min) / (max_response_max - max_response_min + 1e-6)
  
             # 生成动态注意力权
            x_feat_att = self.triple_att(own_response, max_response, local_resp)
           
        
            batch_size = x.indices[:, 0].max() + 1
            selected_diffusion_feats_list = [x.features.clone()]
            selected_diffusion_coords_list = [x.indices.clone()]
            for i in range(batch_size):
                mask = x.indices[:, 0] == i
                valid_num = mask.sum()
                K = int(valid_num * self.diff_scale)
                _, indices = torch.topk(x_feat_att[mask], K)
                selected_coords_copy = x.indices[mask][indices].clone()
                selected_coords_num = selected_coords_copy.shape[0]
                selected_coords_expand = selected_coords_copy.repeat(diffusion_scale, 1)
                
                # 原模型对扩展体素进行初始化
                selected_feats_expand = x.features[mask][indices].repeat(diffusion_scale, 1) * 0.0

                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 3:4] = (
                            selected_coords_copy[:, 3:4] - coords_shift).clamp(min=0, max=w - 1)
                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 2:3] = (
                            selected_coords_copy[:, 2:3] + coords_shift).clamp(min=0, max=h - 1)
                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 1:2] = (
                            selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 3:4] = (
                            selected_coords_copy[:, 3:4] + coords_shift).clamp(min=0, max=w - 1)
                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 2:3] = (
                            selected_coords_copy[:, 2:3] + coords_shift).clamp(min=0, max=h - 1)
                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 1:2] = (
                            selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                if diffusion_scale == 4:
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 3:4] = (
                        selected_coords_copy[:, 3:4] - coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 2:3] = (
                        selected_coords_copy[:, 2:3] - coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 1:2] = (
                        selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 3:4] = (
                            selected_coords_copy[:, 3:4] + coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 2:3] = (
                            selected_coords_copy[:, 2:3] - coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 1:2] = (
                        selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                selected_diffusion_coords_list.append(selected_coords_expand)
                selected_diffusion_feats_list.append(selected_feats_expand)

            coords = torch.cat(selected_diffusion_coords_list)
            final_diffusion_feats = torch.cat(selected_diffusion_feats_list)
        else:
            coords = x.indices.clone()
            final_diffusion_feats = x.features.clone()

        coords[:, 3:4] = coords[:, 3:4] // down_scale[0]
        coords[:, 2:3] = coords[:, 2:3] // down_scale[1]
        coords[:, 1:2] = coords[:, 1:2] // down_scale[2]

        scale_xyz = (x.spatial_shape[0] // down_scale[2]) * (x.spatial_shape[1] // down_scale[1]) * (
                x.spatial_shape[2] // down_scale[0])
        scale_yz = (x.spatial_shape[0] // down_scale[2]) * (x.spatial_shape[1] // down_scale[1])
        scale_z = (x.spatial_shape[0] // down_scale[2])

        merge_coords = coords[:, 0].int() * scale_xyz + coords[:, 3] * scale_yz + coords[:, 2] * scale_z + coords[:, 1]

        features_expand = final_diffusion_feats

        new_sparse_shape = [math.ceil(x.spatial_shape[i] / down_scale[2 - i]) for i in range(3)]
        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True, return_counts=False, dim=0)

        x_merge = torch_scatter.scatter_add(features_expand, unq_inv, dim=0)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // scale_xyz,
                                    (unq_coords % scale_xyz) // scale_yz,
                                    (unq_coords % scale_yz) // scale_z,
                                    unq_coords % scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        x_merge = self.norm(x_merge)

        x_merge = spconv.SparseConvTensor(
            features=x_merge,
            indices=voxel_coords.int(),
            spatial_shape=new_sparse_shape,
            batch_size=x.batch_size
        )
        return x_merge, unq_inv


class PatchExpanding3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, up_x, unq_inv):
        # z, y, x
        n, c = x.features.shape

        x_copy = torch.gather(x.features, 0, unq_inv.unsqueeze(1).repeat(1, c))
        up_x = up_x.replace_feature(up_x.features + x_copy)
        return up_x


LinearOperatorMap = {
    'Mamba': MambaBlock,
    'RWKV': RWKVBlock,
    'RetNet': RetNetBlock,
    'xLSTM': xLSTM_Block,
    'TTT': TTTBlock,
}


class LIONLayer(nn.Module):
    def __init__(self, dim, nums, window_shape, group_size, direction, shift, operator=None, layer_id=0, n_layer=0):
        super(LIONLayer, self).__init__()

        self.window_shape = window_shape
        self.group_size = group_size
        self.dim = dim
        self.direction = direction

        operator_cfg = operator.CFG
        operator_cfg['d_model'] = dim

        block_list = []
        for i in range(len(direction)):
            operator_cfg['layer_id'] = i + layer_id
            operator_cfg['n_layer'] = n_layer
            # operator_cfg['with_cp'] = layer_id >= 16
            operator_cfg['with_cp'] = layer_id >= 0  ## all lion layer use checkpoint to save GPU memory!! (less 24G for training all models!!!)
            print('### use part of checkpoint!!')
            block_list.append(LinearOperatorMap[operator.NAME](**operator_cfg))

        self.blocks = nn.ModuleList(block_list)
        self.window_partition = FlattenedWindowMapping(self.window_shape, self.group_size, shift)
        

    def forward(self, x):
        mappings = self.window_partition(x.indices, x.batch_size, x.spatial_shape)

        for i, block in enumerate(self.blocks):
            indices = mappings[self.direction[i]]
            x_features = x.features[indices][mappings["flat2win"]]
            x_features = x_features.view(-1, self.group_size, x.features.shape[-1])
            

            x_features = block(x_features)

            x.features[indices] = x_features.view(-1, x_features.shape[-1])[mappings["win2flat"]]

        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class TripleAtt(nn.Module):
    def __init__(self, feat_dim, reduction=16):
        super().__init__()
        self.feat_dim = feat_dim
        
        # CoordAtt风格的双路径设计
        self.shared_mlp = nn.Sequential(
            nn.Linear(3, max(8, feat_dim//reduction)),
            nn.BatchNorm1d(max(8, feat_dim//reduction)),
            h_swish()
        )
        
        # 独立权重分支
        self.att_weights = nn.Sequential(
            nn.Linear(max(8, feat_dim//reduction), 3),
            nn.Softmax(dim=1)
        )
        
        # 残差连接参数（保持维度兼容）
        self.alpha = nn.Parameter(torch.ones(1))
        
        # 初始化均值权重更高
        with torch.no_grad():
            self.att_weights[0].weight.data[:,0] *= 1.2  # 强化均值通道
            self.att_weights[0].bias.data = torch.tensor([0.4, 0.3, 0.3])

    def forward(self, mean_feat, max_feat, density_feat):
        # 保持原有维度 [N,3]
        combined = torch.stack([mean_feat, max_feat, density_feat], dim=1)
        
        # 轻量级特征交互
        shared = self.shared_mlp(combined)  # [N, C]
        
        # 生成动态权重
        weights = self.att_weights(shared)  # [N,3]
        
        # 残差增强的加权融合
        return self.alpha * (weights[:,0]*mean_feat + weights[:,1]*max_feat + weights[:,2]*density_feat) + \
               (1-self.alpha) * combined.mean(dim=1)  # 保持维度[N]

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, input_channel, num_pos_feats):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats)
        )

    def forward(self, xyz):
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding

# 定义相对位置编码模块
class RelativePositionEmbeddingLearned(nn.Module):
    """
    Relative position embedding, learned.
    """
    def __init__(self, input_channel, num_pos_feats):
        super().__init__()
        self.relative_position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats)
        )

    def forward(self, relative_xyz):
        relative_position_embedding = self.relative_position_embedding_head(relative_xyz)
        return relative_position_embedding

# 修改后的 LIONBlock 类
class LIONBlock(nn.Module):
    def __init__(self, dim: int, depth: int, down_scales: list, window_shape, group_size, direction, shift=False,
                 operator=None, layer_id=0, n_layer=0, use_relative_pos=False):
        super().__init__()

        self.use_relative_pos = use_relative_pos
        self.down_scales = down_scales

        self.encoder = nn.ModuleList()
        self.downsample_list = nn.ModuleList()
        self.pos_emb_list = nn.ModuleList()
        if self.use_relative_pos:
            self.rel_pos_emb_list = nn.ModuleList()
            # 定义可学习的参数alpha_logit和beta_logit
            self.alpha_logit = nn.Parameter(torch.zeros(1))
            self.beta_logit = nn.Parameter(torch.zeros(1))

        norm_fn = partial(nn.LayerNorm)

        shift = [False, shift]
        for idx in range(depth):
            self.encoder.append(LIONLayer(dim, 1, window_shape, group_size, direction, shift[idx], operator, layer_id + idx * 2, n_layer))
            self.pos_emb_list.append(PositionEmbeddingLearned(input_channel=3, num_pos_feats=dim))
            if self.use_relative_pos:
                self.rel_pos_emb_list.append(RelativePositionEmbeddingLearned(input_channel=3, num_pos_feats=dim))
            self.downsample_list.append(PatchMerging3D(dim, dim, down_scale=down_scales[idx], norm_layer=norm_fn))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        self.upsample_list = nn.ModuleList()
        for idx in range(depth):
            self.decoder.append(LIONLayer(dim, 1, window_shape, group_size, direction, shift[idx], operator, layer_id + 2 * (idx + depth), n_layer))
            self.decoder_norm.append(norm_fn(dim))
            self.upsample_list.append(PatchExpanding3D(dim))

    def forward(self, x):
        features = []
        index = []

        if self.use_relative_pos:
            # 计算softmax以确保alpha和beta的和为1
            weights = torch.softmax(torch.stack([self.alpha_logit, self.beta_logit]), dim=0)
            alpha, beta = weights[0], weights[1]

        for idx, enc in enumerate(self.encoder):
            # 计算绝对位置嵌入
            abs_pos_emb = self.get_pos_embed(
                spatial_shape=x.spatial_shape,
                coors=x.indices[:, 1:],
                embed_layer=self.pos_emb_list[idx]
            )

            if self.use_relative_pos:
                # 计算相对位置嵌入
                rel_pos_emb = self.get_rel_pos_embed(
                    spatial_shape=x.spatial_shape,
                    coors=x.indices[:, 1:],
                    embed_layer=self.rel_pos_emb_list[idx]
                )

                # 加权相加绝对和相对位置嵌入
                combined_pos_emb = 0.7 * abs_pos_emb + 0.3 * rel_pos_emb
            else:
                # 仅使用绝对位置嵌入
                combined_pos_emb = abs_pos_emb

            # 将位置嵌入添加到特征中
            x = replace_feature(x, combined_pos_emb + x.features)
            x = enc(x)
            features.append(x)
            x, unq_inv = self.downsample_list[idx](x)
            index.append(unq_inv)

        i = 0
        for dec, norm, up_x, unq_inv, up_scale in zip(self.decoder, self.decoder_norm, features[::-1],
                                                      index[::-1], self.down_scales[::-1]):
            x = dec(x)
            x = self.upsample_list[i](x, up_x, unq_inv)
            x = replace_feature(x, norm(x.features))
            i = i + 1
        return x

    def get_pos_embed(self, spatial_shape, coors, embed_layer, normalize_pos=True):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = spatial_shape[::-1]  # spatial_shape:   win_z, win_y, win_x ---> win_x, win_y, win_z

        embed_layer = embed_layer
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        z, y, x = coors[:, 0] - win_z / 2, coors[:, 1] - win_y / 2, coors[:, 2] - win_x / 2

        if normalize_pos:
            x = x / win_x * 2 * math.pi  # [-pi, pi]
            y = y / win_y * 2 * math.pi  # [-pi, pi]
            z = z / win_z * 2 * math.pi  # [-pi, pi]

        if ndim == 2:
            location = torch.stack((x, y), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location)

        return pos_embed

    def get_rel_pos_embed(self, spatial_shape, coors, embed_layer, normalize_pos=True):
        '''
        计算相对位置嵌入
        Args:
        coors: shape=[N, 3], order: z, y, x
        '''
        # 相对位置是相对于窗口中心的位置
        window_shape = spatial_shape[::-1]

        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        # 计算窗口中心
        center_z, center_y, center_x = (win_z / 2, win_y / 2, win_x / 2)

        # 计算相对位置
        rel_z = coors[:, 0] - center_z
        rel_y = coors[:, 1] - center_y
        rel_x = coors[:, 2] - center_x

        if normalize_pos:
            rel_x = rel_x / win_x * 2 * math.pi
            rel_y = rel_y / win_y * 2 * math.pi
            rel_z = rel_z / win_z * 2 * math.pi if ndim == 3 else torch.zeros_like(rel_x)

        if ndim == 2:
            rel_location = torch.stack((rel_x, rel_y), dim=-1)
        else:
            rel_location = torch.stack((rel_x, rel_y, rel_z), dim=-1)
        rel_pos_embed = embed_layer(rel_location)

        return rel_pos_embed
    

class MLPBlock(nn.Module):
    def __init__(self, input_channel, out_channel, norm_fn):
        super().__init__()
        self.mlp_layer = nn.Sequential(
            nn.Linear(input_channel, out_channel),
            norm_fn(out_channel),
            nn.GELU())

    def forward(self, x):
        mpl_feats = self.mlp_layer(x)
        return mpl_feats



# for waymo and nuscenes, kitti, once
class LION3DBackboneOneStride(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg

        self.sparse_shape = grid_size[::-1]  # + [1, 0, 0]
        norm_fn = partial(nn.LayerNorm)

        dim = model_cfg.FEATURE_DIM
        num_layers = model_cfg.NUM_LAYERS
        depths = model_cfg.DEPTHS
        layer_down_scales = model_cfg.LAYER_DOWN_SCALES
        direction = model_cfg.DIRECTION
        diffusion = model_cfg.DIFFUSION
        shift = model_cfg.SHIFT
        diff_scale = model_cfg.DIFF_SCALE
        self.window_shape = model_cfg.WINDOW_SHAPE
        self.group_size = model_cfg.GROUP_SIZE
        self.layer_dim = model_cfg.LAYER_DIM
        self.linear_operator = model_cfg.OPERATOR

        self.n_layer = len(depths) * depths[0] * 2 * 2 + 2

        down_scale_list = [[2, 2, 2],
                           [2, 2, 2],
                           [2, 2, 1],
                           [1, 1, 2],
                           [1, 1, 2]
                           ]
        total_down_scale_list = [down_scale_list[0]]
        for i in range(len(down_scale_list) - 1):
            tmp_dow_scale = [x * y for x, y in zip(total_down_scale_list[i], down_scale_list[i + 1])]
            total_down_scale_list.append(tmp_dow_scale)

        assert num_layers == len(depths)
        assert len(layer_down_scales) == len(depths)
        assert len(layer_down_scales[0]) == depths[0]
        assert len(self.layer_dim) == len(depths)


        # linear_1 和 linear_2 使用 combined position embeddings
        self.linear_1 = LIONBlock(
            self.layer_dim[0], depths[0], layer_down_scales[0], self.window_shape[0],
            self.group_size[0], direction, shift=shift, operator=self.linear_operator, layer_id=0, n_layer=self.n_layer,
            use_relative_pos=True  # 启用相对位置编码
        )  ## [27, 27, 32] --> [13, 13, 32]

        self.dow1 = PatchMerging3D(
            self.layer_dim[0], self.layer_dim[0], down_scale=[1, 1, 2],
            norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale
        )

        # linear_2 使用 combined position embeddings
        self.linear_2 = LIONBlock(
            self.layer_dim[1], depths[1], layer_down_scales[1], self.window_shape[1],
            self.group_size[1], direction, shift=shift, operator=self.linear_operator, layer_id=8, n_layer=self.n_layer,
            use_relative_pos=False  # 不启用相对位置编码
        )

        self.dow2 = PatchMerging3D(
            self.layer_dim[1], self.layer_dim[1], down_scale=[1, 1, 2],
            norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale
        )

        # linear_3 不使用相对位置编码
        self.linear_3 = LIONBlock(
            self.layer_dim[2], depths[2], layer_down_scales[2], self.window_shape[2],
            self.group_size[2], direction, shift=shift, operator=self.linear_operator, layer_id=16, n_layer=self.n_layer,
            use_relative_pos=False  # 不启用相对位置编码
        )

        self.dow3 = PatchMerging3D(
            self.layer_dim[2], self.layer_dim[3], down_scale=[1, 1, 2],
            norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale
        )

        # linear_4 不使用相对位置编码
        self.linear_4 = LIONBlock(
            self.layer_dim[3], depths[3], layer_down_scales[3], self.window_shape[3],
            self.group_size[3], direction, shift=shift, operator=self.linear_operator, layer_id=24, n_layer=self.n_layer,
            use_relative_pos=False  # 不启用相对位置编码
        )

        self.dow4 = PatchMerging3D(
            self.layer_dim[3], self.layer_dim[3], down_scale=[1, 1, 2],
            norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale
        )

        self.linear_out = LIONLayer(
            self.layer_dim[3], 1, [13, 13, 2], 256, direction=['x', 'y'], shift=shift,
            operator=self.linear_operator, layer_id=32, n_layer=self.n_layer
        )

        self.num_point_features = dim

        self.backbone_channels = {
            'x_conv1': 128,
            'x_conv2': 128,
            'x_conv3': 128,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.linear_1(x)
        x1, _ = self.dow1(x)  ## 14.0k --> 16.9k  [32, 1000, 1000]-->[16, 1000, 1000]
        x = self.linear_2(x1)
        x2, _ = self.dow2(x)  ## 16.9k --> 18.8k  [16, 1000, 1000]-->[8, 1000, 1000]
        x = self.linear_3(x2)
        x3, _ = self.dow3(x)   ## 18.8k --> 19.1k  [8, 1000, 1000]-->[4, 1000, 1000]
        x = self.linear_4(x3)
        x4, _ = self.dow4(x)  ## 19.1k --> 18.5k  [4, 1000, 1000]-->[2, 1000, 1000]
        x = self.linear_out(x4)

        batch_dict.update({
            'encoded_spconv_tensor': x,
            'encoded_spconv_tensor_stride': 1
        })

        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x1,
                'x_conv2': x2,
                'x_conv3': x3,
                'x_conv4': x4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': torch.tensor([1,1,2], device=x1.features.device).float(),
                'x_conv2': torch.tensor([1,1,4], device=x1.features.device).float(),
                'x_conv3': torch.tensor([1,1,8], device=x1.features.device).float(),
                'x_conv4': torch.tensor([1,1,16], device=x1.features.device).float(),
            }
        })

        return batch_dict


# for argoverse
class LION3DBackboneOneStride_Sparse(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg

        self.sparse_shape = grid_size[::-1]  # + [1, 0, 0]
        norm_fn = partial(nn.LayerNorm)

        dim = model_cfg.FEATURE_DIM
        num_layers = model_cfg.NUM_LAYERS
        depths = model_cfg.DEPTHS
        layer_down_scales = model_cfg.LAYER_DOWN_SCALES
        direction = model_cfg.DIRECTION
        diffusion = model_cfg.DIFFUSION
        shift = model_cfg.SHIFT
        diff_scale = model_cfg.DIFF_SCALE
        self.window_shape = model_cfg.WINDOW_SHAPE
        self.group_size = model_cfg.GROUP_SIZE
        self.layer_dim = model_cfg.LAYER_DIM
        self.linear_operator = model_cfg.OPERATOR

        self.n_layer = len(depths) * depths[0] * 2 * 2 + 2 + 2 * 3

        down_scale_list = [[2, 2, 2],
                           [2, 2, 2],
                           [2, 2, 1],
                           [1, 1, 2],
                           [1, 1, 2]
                           ]
        total_down_scale_list = [down_scale_list[0]]
        for i in range(len(down_scale_list) - 1):
            tmp_dow_scale = [x * y for x, y in zip(total_down_scale_list[i], down_scale_list[i + 1])]
            total_down_scale_list.append(tmp_dow_scale)

        assert num_layers == len(depths)
        assert len(layer_down_scales) == len(depths)
        assert len(layer_down_scales[0]) == depths[0]
        assert len(self.layer_dim) == len(depths)


        # linear_1 和 linear_2 使用 combined position embeddings
        self.linear_1 = LIONBlock(
            self.layer_dim[0], depths[0], layer_down_scales[0], self.window_shape[0],
            self.group_size[0], direction, shift=shift, operator=self.linear_operator, layer_id=0, n_layer=self.n_layer,
            use_relative_pos=True  # 启用相对位置编码
        )  ## [27, 27, 32] --> [13, 13, 32]

        self.dow1 = PatchMerging3D(
            self.layer_dim[0], self.layer_dim[0], down_scale=[1, 1, 2],
            norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale
        )

        # linear_2 使用 combined position embeddings
        self.linear_2 = LIONBlock(
            self.layer_dim[1], depths[1], layer_down_scales[1], self.window_shape[1],
            self.group_size[1], direction, shift=shift, operator=self.linear_operator, layer_id=8, n_layer=self.n_layer,
            use_relative_pos=True  # 启用相对位置编码
        )

        self.dow2 = PatchMerging3D(
            self.layer_dim[1], self.layer_dim[1], down_scale=[1, 1, 2],
            norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale
        )

        # linear_3 不使用相对位置编码
        self.linear_3 = LIONBlock(
            self.layer_dim[2], depths[2], layer_down_scales[2], self.window_shape[2],
            self.group_size[2], direction, shift=shift, operator=self.linear_operator, layer_id=16, n_layer=self.n_layer,
            use_relative_pos=False  # 不启用相对位置编码
        )

        self.dow3 = PatchMerging3D(
            self.layer_dim[2], self.layer_dim[3], down_scale=[1, 1, 2],
            norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale
        )

        # linear_4 不使用相对位置编码
        self.linear_4 = LIONBlock(
            self.layer_dim[3], depths[3], layer_down_scales[3], self.window_shape[3],
            self.group_size[3], direction, shift=shift, operator=self.linear_operator, layer_id=24, n_layer=self.n_layer,
            use_relative_pos=False  # 不启用相对位置编码
        )

        self.dow4 = PatchMerging3D(
            self.layer_dim[3], self.layer_dim[3], down_scale=[1, 1, 2],
            norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale
        )

        self.linear_out = LIONLayer(
            self.layer_dim[3], 1, [13, 13, 2], 256, direction=['x', 'y'], shift=shift,
            operator=self.linear_operator, layer_id=32, n_layer=self.n_layer
        )

        self.dow_out = PatchMerging3D(
            self.layer_dim[3], self.layer_dim[3], down_scale=[1, 1, 2],
            norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale
        )

        # BEV Layers
        self.linear_bev1 = LIONLayer(
            self.layer_dim[3], 1, [25, 25, 1], 512, direction=['x', 'y'], shift=shift,
            operator=self.linear_operator, layer_id=34, n_layer=self.n_layer
        )
        self.linear_bev2 = LIONLayer(
            self.layer_dim[3], 1, [37, 37, 1], 1024, direction=['x', 'y'], shift=shift,
            operator=self.linear_operator, layer_id=36, n_layer=self.n_layer
        )
        self.linear_bev3 = LIONLayer(
            self.layer_dim[3], 1, [51, 51, 1], 2048, direction=['x', 'y'], shift=shift,
            operator=self.linear_operator, layer_id=38, n_layer=self.n_layer
        )

        self.num_point_features = dim

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.linear_1(x)
        x1, _ = self.dow1(x)  ## 14.0k --> 16.9k  [32, 1000, 1000]-->[16, 1000, 1000]
        x = self.linear_2(x1)
        x2, _ = self.dow2(x)  ## 16.9k --> 18.8k  [16, 1000, 1000]-->[8, 1000, 1000]
        x = self.linear_3(x2)
        x3, _ = self.dow3(x)   ## 18.8k --> 19.1k  [8, 1000, 1000]-->[4, 1000, 1000]
        x = self.linear_4(x3)
        x4, _ = self.dow4(x)  ## 19.1k --> 18.5k  [4, 1000, 1000]-->[2, 1000, 1000]
        x = self.linear_out(x4)

        x, _ = self.dow_out(x)

        x = self.linear_bev1(x)
        x = self.linear_bev2(x)
        x = self.linear_bev3(x)

        x_new = spconv.SparseConvTensor(
            features=x.features,
            indices=x.indices[:, [0, 2, 3]].type(torch.int32),  # x.indices,
            spatial_shape=x.spatial_shape[1:],
            batch_size=x.batch_size
        )

        batch_dict.update({
            'encoded_spconv_tensor': x_new,
            'encoded_spconv_tensor_stride': 1
        })

        batch_dict.update({'spatial_features_2d': x_new})

        return batch_dict
        