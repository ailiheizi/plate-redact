# -*- coding: UTF-8 -*-
"""
轻量多目标跟踪器 (ByteTrack 风格, 纯 numpy 实现)

- 两级关联: 高置信度检测优先匹配, 低置信度检测补充
- 匀速运动预测, 平滑框位置
- 支持短暂丢失保持 (max_age), 场景切换重置

用于车牌打码场景: 检测失败/遮挡/运动模糊时, 继续跟随已确认的车牌框,
做到「时间冗余」——不依赖单帧检测成败, 而是靠轨迹连续性保证每一帧都被遮挡。
"""
import numpy as np


def iou_matrix(tracks, dets):
    """tracks(ltrb) x dets(ltrb) 的 IoU 矩阵"""
    if len(tracks) == 0 or len(dets) == 0:
        return np.zeros((len(tracks), len(dets)))
    t = np.array(tracks, dtype=float)
    d = np.array(dets, dtype=float)
    area_t = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    area_d = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
    ix1 = np.maximum(t[:, None, 0], d[None, :, 0])
    iy1 = np.maximum(t[:, None, 1], d[None, :, 1])
    ix2 = np.minimum(t[:, None, 2], d[None, :, 2])
    iy2 = np.minimum(t[:, None, 3], d[None, :, 3])
    iw = np.maximum(0, ix2 - ix1)
    ih = np.maximum(0, iy2 - iy1)
    inter = iw * ih
    union = area_t[:, None] + area_d[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-6), 0)


def greedy_match(cost):
    """按分数降序贪心匹配, 返回 (track_idx, det_idx) 列表"""
    if cost.size == 0:
        return []
    matched_t, matched_d = set(), set()
    pairs = []
    idx = np.dstack(np.unravel_index(np.argsort(cost.ravel())[::-1], cost.shape))[0]
    for i, j in idx:
        if i in matched_t or j in matched_d:
            continue
        matched_t.add(i)
        matched_d.add(j)
        pairs.append((int(i), int(j)))
    return pairs


def center_distance_matrix(tracks, dets):
    """中心点距离矩阵(像素), 用于快速移动目标的帧间关联"""
    if len(tracks) == 0 or len(dets) == 0:
        return np.full((len(tracks), len(dets)), 1e9)
    tc = np.array([[(t[0]+t[2])/2, (t[1]+t[3])/2] for t in tracks])
    dc = np.array([[(d[0]+d[2])/2, (d[1]+d[3])/2] for d in dets])
    return np.sqrt(((tc[:, None, :] - dc[None, :, :]) ** 2).sum(axis=2))


class Track:
    def __init__(self, box, track_id, valid, max_speed=0.35, start_frame=0):
        self.id = track_id
        self.valid = valid
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.lost = False
        self.box = [float(v) for v in box]
        self.prev_box = list(self.box)
        self.max_speed = max_speed
        self.w = self.box[2] - self.box[0]
        self.h = self.box[3] - self.box[1]
        # 完整轨迹历史: [(frame_idx, [x1,y1,x2,y2]), ...] (用于往前推回溯)
        self.history = [(start_frame, list(self.box))]

    def predict(self):
        """匀速外推 (限速防跳变)"""
        dx = self.box[0] - self.prev_box[0]
        dy = self.box[1] - self.prev_box[1]
        self.prev_box = list(self.box)
        if abs(dx) > self.w * self.max_speed:
            dx = np.sign(dx) * self.w * self.max_speed
        if abs(dy) > self.h * self.max_speed:
            dy = np.sign(dy) * self.h * self.max_speed
        self.box = [self.box[0] + dx, self.box[1] + dy,
                    self.box[2] + dx, self.box[3] + dy]
        self.age += 1
        self.time_since_update += 1
        return self.box

    def update(self, box, valid, frame_idx=None):
        self.w = (self.box[2] - self.box[0] + box[2] - box[0]) / 2
        self.h = (self.box[3] - self.box[1] + box[3] - box[1]) / 2
        self.box = [float(v) for v in box]
        if valid:
            self.valid = True
        self.hits += 1
        self.time_since_update = 0
        self.lost = False
        if frame_idx is not None:
            self.history.append((frame_idx, list(self.box)))
            # 限制历史长度(防止长时间track内存增长)
            if len(self.history) > 300:
                self.history = self.history[-300:]
        return self.box


class PlateTracker:
    def __init__(self, max_age=30, iou_high=0.4, iou_low=0.2,
                 min_hits=1, conf_low=0.2, conf_high=0.3,
                 dist_high=60, dist_low=120, max_tracks=20):
        self.max_age = max_age
        self.iou_high = iou_high
        self.iou_low = iou_low
        self.min_hits = min_hits
        self.conf_low = conf_low
        self.conf_high = conf_high
        self.dist_high = dist_high
        self.dist_low = dist_low
        self.max_tracks = max_tracks
        self.tracks = []
        self.next_id = 0
        self.all_history = []  # 所有track的历史(删除时保存, 供往前推回溯)

    def reset(self):
        self.tracks = []
        self.next_id = 0

    def predict(self):
        for t in self.tracks:
            t.predict()

    def update(self, dets, frame_idx=None):
        """dets: [(x1,y1,x2,y2,conf,valid), ...] 按置信度排序由调用方保证
        返回当前存活 track: [(box, track_id, valid), ...]
        """
        if len(dets) == 0:
            self._remove_stale()
            return self.active_tracks()
        confs = np.array([d[4] for d in dets])
        high_idx = set(np.where(confs >= self.conf_high)[0])
        low_idx = set(np.where((confs >= self.conf_low) & (confs < self.conf_high))[0])

        track_boxes = [t.box for t in self.tracks]
        cost = iou_matrix(track_boxes, [d[:4] for d in dets])
        dist = center_distance_matrix(track_boxes, [d[:4] for d in dets])

        matched_tracks = set()
        matched_dets = set()

        # 高置信度: IoU 关联, 或中心距离近(快速移动)也关联
        for ti, di in greedy_match(cost):
            if ti in matched_tracks or di in matched_dets:
                continue
            if di in high_idx:
                ok_match = cost[ti, di] >= self.iou_high or dist[ti, di] <= self.dist_high
                if ok_match:
                    self.tracks[ti].update(dets[di][:4], dets[di][5], frame_idx)
                    matched_tracks.add(ti)
                    matched_dets.add(di)
        # 低置信度: 更宽松的距离关联
        for ti, di in greedy_match(cost):
            if ti in matched_tracks or di in matched_dets:
                continue
            if di in low_idx:
                ok_match = cost[ti, di] >= self.iou_low or dist[ti, di] <= self.dist_low
                if ok_match:
                    self.tracks[ti].update(dets[di][:4], dets[di][5], frame_idx)
                    matched_tracks.add(ti)
                    matched_dets.add(di)

        for di in high_idx - matched_dets:
            box = [float(v) for v in dets[di][:4]]
            self.tracks.append(Track(box, self.next_id, dets[di][5], start_frame=frame_idx or 0))
            self.next_id += 1

        self._remove_stale()
        return self.active_tracks()

    def active_tracks(self, max_stale=6):
        """返回可打码的 track: 已确认(valid)且近期被检测更新过
        max_stale: 超过 N 帧未更新的 track 不再输出(防僵尸框残留)
        """
        return [(list(t.box), t.id, t.valid)
                for t in self.tracks
                if t.hits >= self.min_hits
                and t.valid
                and t.time_since_update <= max_stale]

    def _remove_stale(self):
        # 删除前保存 history(往前推回溯用)
        removed = [t for t in self.tracks
                   if t.time_since_update > self.max_age]
        for t in removed:
            if len(getattr(t, 'history', [])) >= 2:
                self.all_history.append(t.history)
        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.max_age]
        # 限制总 track 数, 防止误检累积 (隐私优先可调大, 车多时被截断会漏打)
        if len(self.tracks) > self.max_tracks:
            kept = sorted(self.tracks, key=lambda t: t.time_since_update)[:self.max_tracks]
            dropped = [t for t in self.tracks if t not in kept]
            # 被截断的 track 也保存历史, 供往前推回溯使用
            for t in dropped:
                if len(getattr(t, 'history', [])) >= 2:
                    self.all_history.append(t.history)
            self.tracks = kept
