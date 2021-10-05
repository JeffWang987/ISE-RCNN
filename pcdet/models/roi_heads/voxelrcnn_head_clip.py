import torch
import torch.nn as nn
import torch.nn.functional as F
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ...utils import common_utils, box_utils
from .roi_head_template import RoIHeadTemplate
from clip_related import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import numpy as np

class VoxelRCNNHeadClip(RoIHeadTemplate):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.crop_method = self.model_cfg.CLIP.CROP_METHOD
        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [backbone_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg[src_name].NSAMPLE,
                radii=LAYER_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg[src_name].POOL_METHOD,
            )
            
            self.roi_grid_pool_layers.append(pool_layer)

            c_out += sum([x[-1] for x in mlps])

        # ViT channel
        c_out += 512
        self.point_wise_in_channel = c_out


        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # c_out = sum([x[-1] for x in mlps])
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        cls_fc_list = []
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.CLS_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.CLS_FC[k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Conv1d(pre_channel, self.num_class, kernel_size=1, bias=True)

        reg_fc_list = []
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.REG_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Conv1d(pre_channel, self.box_coder.code_size * self.num_class, kernel_size=1, bias=True)

        self.point_pool = nn.MaxPool1d(216)
        self.point_wise_layers = nn.Sequential(
            nn.Linear(self.point_wise_in_channel, self.point_wise_in_channel),
            nn.ReLU()
        )
        self.channel_pool = nn.MaxPool1d(self.point_wise_in_channel)
        self.channel_wise_layers = nn.Sequential(
            nn.Linear(216, 216),
            nn.ReLU()
        )
        self.init_weights()

        self.vit, _ = clip.load("ViT-B/32", device="cuda", jit=False)
        self.vit_preprocess = Compose([
            Resize((224,224), interpolation=Image.BICUBIC),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        # TODO not converge
        if self.model_cfg.CLIP.FREEZE:
            for param in self.vit.parameters():
                param.requires_grad = False

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    
        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)
    
    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        rois = batch_dict['rois']
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)
        
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)  

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            if with_vf_transform:
                cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
            else:
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            # compute voxel center xyz and batch_cnt
            cur_coords = cur_sp_tensors.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            # get voxel2point tensor
            v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)
            # compute the grid coordinates in this scale, in [batch_idx, x y z] order
            cur_roi_grid_coords = roi_grid_coords // cur_stride
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()
            # voxel neighbor aggregation
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                new_xyz_batch_cnt=roi_grid_batch_cnt,
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                features=cur_sp_tensors.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor
            )

            pooled_features = pooled_features.view(
                -1, self.pool_cfg.GRID_SIZE ** 3,
                pooled_features.shape[-1]
            )  # (BxN, 6x6x6, C)
            pooled_features_list.append(pooled_features)
        
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)
        
        return ms_pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points
    
    def roi3d_2d(self, rois, trans_lidar_to_cam, trans_cam_to_img, random_world_rotation, random_world_flip_enable, noise_scale):
        # 3D roi-> 8 corner coordinates
        batch_size = rois.shape[0]
        corners = box_utils.boxes_to_corners_3d(rois.reshape(-1, 7))  # BxN_rois, 8, 3
        corners = corners.reshape(batch_size, -1, 8, 3)  # B, N_rois, 8, 3

        # according to augmentation sequence, inverse the coordinates
        corners = corners / noise_scale[:,None,None,None]  # 1/sclae
        corners = common_utils.rotate_points_along_z(corners.reshape(batch_size, -1, 3), -random_world_rotation)  # rot.inverse

        for i, flip_enable in enumerate(random_world_flip_enable):
            if flip_enable:
                corners[i, :, 1] = - corners[i, :, 1]  # flip back
        
        # transfer to image view
        homo_corners = torch.cat([corners, corners.new_ones([*corners.shape[:2], 1])], dim=2)
        img_corners = homo_corners@trans_lidar_to_cam.permute(0,2,1)@trans_cam_to_img.permute(0,2,1)
        img_corners = img_corners[..., :2] / (img_corners[..., 2:3]+1e-9)
        img_corners = img_corners.reshape(batch_size, -1, 8, 2)  # [B, 128, 8, 2]

        # 3D corners -> 2D corners
        patch_top_value, patch_top_indices = torch.min(img_corners[..., 1], dim=2)
        patch_bottom_value, patch_bottom_indices = torch.max(img_corners[..., 1], dim=2)
        patch_left_value, patch_left_indices = torch.min(img_corners[..., 0], dim=2)
        patch_right_value, patch_right_indices = torch.max(img_corners[..., 0], dim=2)

        return torch.stack([patch_top_value, patch_bottom_value, patch_left_value, patch_right_value], dim=2), img_corners  # [B, 128, 4]

    def crop_image_from_roi(self, images, image_rois, image_pads):
        # images [B, 3, H, W]
        # image_rois [B, N_rois, 4]

        def check_boundary(top, bottom, left, right, H, W):
            top = top if top>0 else 0
            bottom = bottom if bottom>0 else 0
            left = left if left>0 else 0
            right = right if right>0 else 0

            top = top if top<H else H-1
            bottom = bottom if bottom<H else H-1
            left = left if left<W else W-1
            right = right if right<W else W-1

            if top == bottom:
                if top == 0:
                    bottom += 50
                else:
                    top -= 50
            if left == right:
                if left == 0:
                    right += 50
                else:
                    left -= 50
            return top, bottom, left, right

        image_patches = []
        for i in range(image_rois.shape[0]):
            image = images[i]
            image_pad = (image_pads[i].to(torch.int))
            image = image[:, image_pad[0,0]: image.shape[1]-image_pad[0,1], image_pad[1,0]: image.shape[2]-image_pad[1,1]]
            for j in range(image_rois.shape[1]):
                top, bottom, left, right = image_rois[i, j].long()
                top, bottom, left, right = check_boundary(top, bottom, left, right, image.shape[1], image.shape[2])
                this_crop = self.vit_preprocess(image[:, top:bottom, left:right])
                image_patches.append(this_crop)
        image_patches = torch.cat(image_patches).reshape(*image_rois.shape[:2], *this_crop.shape[-3:])
        # [B, Nroi, 3,224,224]
        return image_patches

    def clip_embedding(self, image_patches):
        # [B, N_patches, 3, 224, 224]
        batch_size, N_patches = image_patches.shape[0], image_patches.shape[1]
        all_embeddings = self.vit.encode_image(image_patches.reshape(-1, 3, 224, 224))
        patch_embedding = all_embeddings.reshape(batch_size, N_patches, 512)
        return patch_embedding

    def perspective_channel_attention(self, roi_feat, vit_feat):
        # roi_feat (BxN, C1, 6, 6, 6)
        # vit_feat [B, N, C2]

        # simple cat  
        roi_feat = roi_feat.reshape(roi_feat.shape[0], roi_feat.shape[1], -1)  # [BxN, C1, 216]
        vit_feat = vit_feat.reshape(-1, vit_feat.shape[-1], 1).expand(roi_feat.shape[0], vit_feat.shape[-1], roi_feat.shape[-1])  # [BxN, C2, 216]
        shared_feat = torch.cat([roi_feat, vit_feat], dim=1)  # [BxN, C1+C2, 216]

        # perspective_channel_attention
        # point wise max pool
        p_feat = self.point_pool(shared_feat).squeeze(2)  # [BxN, C1+C2]
        p_feat = self.point_wise_layers(p_feat).unsqueeze(2)  # [BxN, C1+C2, 1]

        # channel wise max pool
        c_feat = self.channel_pool(shared_feat.permute(0,2,1)).squeeze(2)  # [BxN, 216]
        c_feat = self.channel_wise_layers(c_feat).unsqueeze(1)  # [BxN, 1, 216]

        attention_map = torch.sigmoid(p_feat@c_feat)  # [BxN, C1+C2, 216]

        return shared_feat * attention_map  # [BxN, C1+C2, 216]

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # clip
        images = batch_dict['images']  # [B,3,H,W]
        image_pads = batch_dict['image_pads']  # [B,2,2]
        rois = targets_dict['rois']  # [B,128,7]
        trans_lidar_to_cam = batch_dict['trans_lidar_to_cam']
        trans_cam_to_img = batch_dict['trans_cam_to_img']
        random_world_rotation = batch_dict['random_world_rotation'] if self.training else torch.zeros([rois.shape[0]]).cuda()
        random_world_flip_enable = batch_dict['random_world_flip_enable'] if self.training else torch.zeros([rois.shape[0]]).cuda()
        noise_scale = batch_dict['noise_scale'] if self.training else torch.ones([rois.shape[0]]).cuda()

        # roi3d -> roi2d
        # image_rois [B, Nroi, 4]
        # roi_map [B, Nroi], 0~Nroi
        with torch.no_grad():
            image_rois, _ = self.roi3d_2d(rois, trans_lidar_to_cam, trans_cam_to_img, random_world_rotation, random_world_flip_enable, noise_scale)
            roi_map = torch.arange(rois.shape[0]* rois.shape[1]).reshape(rois.shape[0], -1)
            # image_patches [B, N_patch,3,224,224]
            if self.crop_method == 1:
                image_patches = self.crop_image_from_roi(images, image_rois, image_pads)
            # elif self.crop_method == 2:
            #     image_patches = self.crop_image_from_roi_v2(images, image_rois, image_pads)
            else:
                raise NotImplementedError
            # patch_embedding [B, N_patch, 512]
        patch_embedding = self.clip_embedding(image_patches)
        # [B, N, C]->[BxN, C]->[Bx128, C]->[B, 128, C]
        roi_embedding = patch_embedding.reshape(-1, patch_embedding.shape[-1])[roi_map.reshape(-1), :].reshape(patch_embedding.shape[0], roi_map.shape[1], -1)
        # RoI aware pooling
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        # perspective channel attention
        shared_features = self.perspective_channel_attention(pooled_features, roi_embedding)  # [BxN, C1+C2, 216]
        shared_features = self.shared_fc_layer(shared_features.view(batch_size_rcnn, -1, 1))

        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features))

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
