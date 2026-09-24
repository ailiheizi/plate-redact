# -*- coding: UTF-8 -*-
"""
视觉大模型车牌兜底检测模块 (Qwen2.5-VL)

用途: 专用车牌检测器(检测为空/低置信)时, 用视觉大模型二次复核"画面里有没有像车牌的区域"。
- 只判断"像不像车牌", 不识别文字 → 不依赖字符集, 跨国家/地区通用
- 分块检测提升召回(整图一次 + 左右分块二次)
- 模型只加载一次(单例), CUDA/MPS/CPU 自动选择
- 输出: 映射回原图坐标的矩形框列表

依赖需自行安装: torch / transformers / pillow (Qwen2.5-VL 权重另行获取,
例如从 ModelScope 下载 Qwen2.5-VL-3B-Instruct)。本仓库不附带任何权重。
"""
import glob
import os
import re
import time

import numpy as np
import cv2


def find_default_model():
    """在常见的 ModelScope 缓存目录里查找 Qwen2.5-VL 权重(找不到返回空串)"""
    cands = glob.glob(os.path.expanduser(
        '~/.cache/modelscope/models/Qwen--Qwen2.5-VL-3B-Instruct/snapshots/*'))
    return sorted(cands)[-1] if cands else ''


class VLPlateDetector:
    _instance = None

    def __init__(self, model_path, device=None, max_pixels=2048):
        import torch
        self.torch = torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.model_path = model_path
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)
        dtype = torch.float16 if device != 'cpu' else torch.float32
        print(f'[LLM] 加载 Qwen2.5-VL 到 {device} ...')
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=256 * 28 * 28, max_pixels=max_pixels * 28 * 28)
        print('[LLM] Qwen2.5-VL 加载完成')

    @classmethod
    def get_instance(cls, model_path, device=None):
        if cls._instance is None:
            cls._instance = cls(model_path, device)
        return cls._instance

    def _ask(self, img_bgr, prompt, max_tokens=150):
        """对一张 BGR 图像提问, 返回模型文本回答"""
        from PIL import Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': pil_img},
            {'type': 'text', 'text': prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_img],
                                return_tensors='pt').to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                      do_sample=False)  # 贪心解码: MPS上采样会inf/nan崩溃
        answer = self.processor.batch_decode(
            out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        return answer

    @staticmethod
    def parse_boxes(answer):
        """从 LLM 回答中提取矩形框 [[x1,y1,x2,y2], ...], 容错解析"""
        boxes = []
        # 1) 优先: 提取方括号/括号包裹的坐标组 [x1,y1,x2,y2]
        for m in re.finditer(r'[\[\{\(]\s*([\d\s.,]+)\s*[\]\}\)]', answer):
            nums = [float(x) for x in re.findall(r'[\d.]+', m.group(1))]
            # 只取 4 个的组 (可能是 x1,y1,x2,y2)
            if len(nums) >= 4:
                boxes.append(nums[:4])
        # 2) bbox_2d / box 字段
        if not boxes:
            for pat in [r'bbox_2d"?\s*:\s*\[([\d\s,\.]+)\]',
                        r'box"?\s*:\s*\[([\d\s,\.]+)\]']:
                for m in re.finditer(pat, answer):
                    nums = [float(x) for x in re.findall(r'[\d.]+', m.group(1))]
                    if len(nums) >= 4:
                        boxes.append(nums[:4])
                if boxes:
                    break
        # 3) 纯数字序列兜底: 裸坐标 "850,426,907,443" 或 "850 426 907 443"
        if not boxes:
            nums = [float(x) for x in re.findall(r'[\d.]+', answer)]
            for i in range(0, len(nums) - 3, 4):
                boxes.append(nums[i:i + 4])
        # 过滤非法框
        valid = []
        for b in boxes:
            x1, y1, x2, y2 = b
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            if x1 > x2 or y1 > y2:
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
            valid.append([x1, y1, x2, y2])
        return valid

    def detect(self, frame, split_blocks=2, verbose=True):
        """检测一张 BGR 帧中的车牌区域, 返回映射回原图坐标的 [[x1,y1,x2,y2],...]
        split_blocks: 整图检测为空时, 将画面水平分为几块分别检测(提升小目标召回)
        """
        H, W = frame.shape[:2]
        prompt = (f'图像尺寸{W}x{H}。检测画面中所有车牌。'
                  f'请列出每个车牌左上角x1,y1和右下角x2,y2的精确整数坐标。'
                  f'格式: [x1,y1,x2,y2]，例如 [850,426,907,443]。'
                  f'只输出坐标数组，多个车牌用逗号分隔，不要解释。如果没有车牌输出 []。')
        t0 = time.time()
        answer = self._ask(frame, prompt)
        boxes = self.parse_boxes(answer)
        if verbose:
            print(f'[LLM] 整图: {len(boxes)} 个 ({time.time()-t0:.0f}s)')

        # 整图未检出时, 分块检测提升召回
        if not boxes and split_blocks > 1:
            for bi in range(split_blocks):
                x1 = bi * W // split_blocks
                x2 = (bi + 1) * W // split_blocks
                blk = frame[:, x1:x2]
                bw = blk.shape[1]
                prompt_b = (f'图像尺寸{bw}x{H}。检测画面中所有车牌。'
                            f'请列出每个车牌左上角x1,y1和右下角x2,y2的精确整数坐标。'
                            f'格式: [x1,y1,x2,y2]，例如 [50,100,200,130]。'
                            f'只输出坐标数组，多个车牌用逗号分隔，不要解释。如果没有车牌输出 []。')
                t0 = time.time()
                ans_b = self._ask(blk, prompt_b)
                blk_boxes = self.parse_boxes(ans_b)
                for b in blk_boxes:
                    b[0] += x1
                    b[2] += x1
                boxes += blk_boxes
                if verbose:
                    print(f'[LLM] 块{bi}: {len(blk_boxes)} 个 ({time.time()-t0:.0f}s)')
        return boxes
