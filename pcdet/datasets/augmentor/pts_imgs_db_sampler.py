import pickle

import numpy as np
import cv2
from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils
import torch


class PtsImgsDataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.collision_thres = self.sampler_cfg.COLLISION_THRES
        self.collision_thres_mixed = True if self.collision_thres==-1 else False
        self.blend_method = self.sampler_cfg.BLEND_METHOD
        self.logger = logger
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []

        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def blend_image(self, image, obj_image, center_image, method=0):
        # method = 0 direct blend
        # method = 1 gaussian blur
        # method = 2 possion blend
        # method = 3 mixed the above 3
        obj_mask = (obj_image[:, :, 0:1] != 0) | (obj_image[:, :, 1:2] != 0) | (obj_image[:, :, 2:3] != 0)  # [H, W, 1]
        obj_H, obj_W = obj_image.shape[:2]
        center_image = [int(round(center_image[0])), int(round(center_image[1]))]  # x,y
        x1, y1 = center_image[0] - obj_W//2, center_image[1] - obj_H//2
        x2, y2 = x1 + obj_W, y1 + obj_H
        cut_x1 = x1 if x1>=1 else 1
        cut_y1 = y1 if y1>=1 else 1
        # cut_x2 = x2 if x2>=0 else 0
        # cut_y2 = y2 if y2>=0 else 0
        # cut_x1 = x1 if x1<image.shape[1] else image.shape[1]-1
        # cut_y1 = y1 if y1<image.shape[0] else image.shape[0]-1
        cut_x2 = x2 if x2<image.shape[1]-1 else image.shape[1]-2
        cut_y2 = y2 if y2<image.shape[0]-1 else image.shape[0]-2
        off_x1 = cut_x1 - x1
        off_x2 = cut_x2 - x2
        off_y1 = cut_y1 - y1
        off_y2 = cut_y2 - y2

        assert off_x1 >= 0 and off_x2 <= 0 and off_y1 >= 0 and off_y2 <= 0
        obj_mask = obj_mask[off_y1:obj_H+off_y2, off_x1:off_x2+obj_W, :]
        img_mask = ~obj_mask
        obj_mask = obj_mask.astype(np.float32)
        img_mask = img_mask.astype(np.float32)
        obj_image = obj_image[off_y1:obj_H+off_y2, off_x1:off_x2+obj_W, :]

        if method == 3:  # mixed:
            method  = np.random.choice([0, 2])
            
        if method == 0:  # direct
            assert obj_image.shape[:2] == image[cut_y1:cut_y2, cut_x1:cut_x2].shape[:2]
            image[cut_y1:cut_y2, cut_x1:cut_x2] = obj_mask * obj_image + img_mask * image[cut_y1:cut_y2, cut_x1:cut_x2]

        elif method == 1:  # gaussian
            raise NotImplementedError

        elif method == 2:  # possion
            assert off_y1*off_y2<=0 and off_x1*off_x2<=0
            center_x = center_image[0] + (off_x1 + off_x2)//2
            center_y = center_image[1] + (off_y1 + off_y2)//2
            try:
                image = cv2.seamlessClone(obj_image.astype(np.uint8), image.astype(np.uint8), (255*np.concatenate([obj_mask,obj_mask,obj_mask],axis=2)).astype(np.uint8), (center_x, center_y), cv2.NORMAL_CLONE)
            except:
                image = image
                print('\n center:{}'.format((center_x, center_y)))
                print('\n obj_mask.shape:{}'.format(obj_mask.shape))
                print('\n obj_image.shape:{}'.format(obj_image.shape))
                print('\n image.shape:{}'.format(image.shape))

        return image.astype(np.float32)

    def put_db_imgs_on_image(self, obj_images_list, image, large_sampled_gt_boxes, trans_lidar_to_cam, trans_cam_to_img):
        # obj_images [Hi, Wi, C],i=1..N
        # image [H, W, C]
        # large_sampled_gt_boxes [N, 7]
        db_obj_lidar_center = large_sampled_gt_boxes[:, :3]
        front_view_distance = db_obj_lidar_center[:, 0]
        _, front_view_distance_idxs = front_view_distance.sort(descending=True)
        center_homo = np.concatenate([db_obj_lidar_center, np.ones([db_obj_lidar_center.shape[0], 1])], axis=1) # N,4
        centers_img = center_homo @ trans_lidar_to_cam.T @ trans_cam_to_img.T  # N, 3
        centers_img = centers_img[:, :2] / centers_img[:, 2:3]

        for distance_idx in front_view_distance_idxs: 
            obj_image = obj_images_list[distance_idx]
            center_img = centers_img[distance_idx]

            image = self.blend_image(image, obj_image, center_img, method=self.blend_method)

        return image
        
    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):
        gt_boxes_mask = data_dict['gt_boxes_mask']
        image = data_dict['images'] * 255
        trans_lidar_to_cam = data_dict['trans_lidar_to_cam']
        trans_cam_to_img = data_dict['trans_cam_to_img']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['points']
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []
        obj_images_list = []
        for idx, info in enumerate(total_valid_sampled_dict):
            file_pts_path = self.root_path / info['path']
            file_img_path = self.root_path / (info['path'].split('.bin')[0] + '_seg_img.bin')
            obj_img = np.fromfile(str(file_img_path), dtype=np.uint8).reshape([*info['seg_bbox'], 3]).astype(np.float32)  # H, W, C
            obj_points = np.fromfile(str(file_pts_path), dtype=np.float32).reshape([-1, self.sampler_cfg.NUM_POINT_FEATURES])
            obj_points[:, :3] += info['box3d_lidar'][:3]

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            obj_points_list.append(obj_points)
            obj_images_list.append(obj_img)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        image = self.put_db_imgs_on_image(obj_images_list, image, large_sampled_gt_boxes, trans_lidar_to_cam, trans_cam_to_img)
        # check visualization
            # import cv2
            # this_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
            # cv2.imwrite('/mnt/cpfs/users/gpuwork/xiaofeng.wang/3ddet/clip/OpenPCDet/debug/vis/vis_images/blend_img_avoid_collision/possion/{}.jpg'.format(data_dict['frame_id']), this_img.astype(np.uint8))
        points = np.concatenate([obj_points, points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        data_dict['images'] = image/255
        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []
        total_obj_2d_centers = []
        total_obj_2d_WH = []
        if self.collision_thres_mixed:
            self.collision_thres = np.random.choice([0.1, 0.3, 0.5, 0.7])
        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)

                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)  # N, 7

                sampled_img_2d_WH = np.stack([x['seg_bbox'] for x in sampled_dict], axis=0).astype(np.float32)  # N, 2
                sampled_img_2d_center = np.concatenate([sampled_boxes[:, :3], np.ones([sampled_boxes.shape[0], 1])], axis=1) @ data_dict['trans_lidar_to_cam'].T @ data_dict['trans_cam_to_img'].T
                sampled_img_2d_center = sampled_img_2d_center[:, :2] / sampled_img_2d_center[:, 2:3]  # N, 2

                # mask1: sampled obj image centers should at this image shape
                mask_center_in_scene = (sampled_img_2d_center[:, 0]>0) & (sampled_img_2d_center[:, 0]<data_dict['images'].shape[1]-1) & \
                    (sampled_img_2d_center[:, 1]>0) & (sampled_img_2d_center[:, 1]<data_dict['images'].shape[0]-1)

                if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                    sampled_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)

                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                lidar_bev_mask = (iou1.max(axis=1) + iou2.max(axis=1)) == 0

                valid_mask = lidar_bev_mask & mask_center_in_scene
                valid_mask = valid_mask.nonzero()[0]

                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]
                total_obj_2d_centers.append(sampled_img_2d_center[valid_mask])
                total_obj_2d_WH.append(sampled_img_2d_WH[valid_mask])

                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)
                
        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        total_obj_2d_centers = np.concatenate(total_obj_2d_centers, axis=0)
        total_obj_2d_WH = np.concatenate(total_obj_2d_WH, axis=0)

        # collision check
        exist_2dbbox = data_dict['gt_boxes2d']
        exist_fake_3d_bbox = np.concatenate([(exist_2dbbox[:,:2] + exist_2dbbox[:,2:])/2, np.ones([exist_2dbbox.shape[0],1]), exist_2dbbox[:,2:] - exist_2dbbox[:,:2], np.zeros([exist_2dbbox.shape[0],2])], axis=1)  # N1,7
        obj_fake_3d_bbox = np.concatenate([total_obj_2d_centers, np.ones([total_obj_2d_centers.shape[0],1]), total_obj_2d_WH, np.zeros([total_obj_2d_centers.shape[0],2])], axis=1) # N2, 7
        all_fake_3d_bbox = np.concatenate([obj_fake_3d_bbox, exist_fake_3d_bbox], axis=0) # N3, 7
        iou_obj_all = iou3d_nms_utils.boxes_bev_iou_cpu(obj_fake_3d_bbox, all_fake_3d_bbox)  # N2, N3
        iou_obj_all[range(obj_fake_3d_bbox.shape[0]), range(obj_fake_3d_bbox.shape[0])] = 0
        all_areas = np.concatenate([total_obj_2d_WH[:,0] * total_obj_2d_WH[:,1], (exist_2dbbox[:,2]-exist_2dbbox[:,0])*(exist_2dbbox[:,3]-exist_2dbbox[:,1])], axis=0)  # N2, N3
        block_ratio = np.zeros([iou_obj_all.shape[0], iou_obj_all.shape[1]])  # N2, N3
        # collision ratio
        all_3d_box = np.concatenate([sampled_gt_boxes, gt_boxes], axis=0)
        for i in range(iou_obj_all.shape[0]):
            for j in range(iou_obj_all.shape[1]):
                if sampled_gt_boxes[i,0] < all_3d_box[j,0]:
                    block_ratio[i,j] = 0
                else:
                    area_block = iou_obj_all[i,j]/(1+iou_obj_all[i,j])*(all_areas[i]+all_areas[j])
                    block_ratio[i,j] = area_block/all_areas[i]
        block_ratio = np.max(block_ratio, axis=1)  # N2
        mask_collision = block_ratio < self.collision_thres  # N2
        sampled_gt_boxes = sampled_gt_boxes[mask_collision]
        total_valid_sampled_dict = [total_valid_sampled_dict[i] for i in mask_collision.nonzero()[0]]

        if total_valid_sampled_dict.__len__() > 0:
            data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict)

        data_dict.pop('gt_boxes_mask')
        data_dict.pop('gt_boxes2d')
        return data_dict
