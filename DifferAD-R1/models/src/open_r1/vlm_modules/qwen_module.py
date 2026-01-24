from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import AutoModelForImageTextToText
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch

from vlm_modules.vlm_module import VLMBaseModule

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        elif "Qwen3-VL" in model_id:
            model_cls = AutoModelForImageTextToText
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return ['max_pixels', 'min_pixels']
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the bounding box in <answer> </answer> tags."
            case "location":
                return "You are an industrial inspector who checks products by images.{Question}"
            case "no":
                return "You are a helpful assistant.{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        single_img_with_cot = kwargs['single_img_with_cot']
        image_path = kwargs['image_path'][0][0]
        is_single_image = 'dataset/coco/train2014/COCO' in image_path
        import re
        import os
        if (not single_img_with_cot) and is_single_image:
            pattern = r"\(\d+,\s*\d+\),\s*\(\d+,\s*\d+\)"
        else:
            pattern = r"<think>.*?</think>\s*<answer>\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)</answer>$"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- Format reward: {[1.0 if match else 0.0 for match in matches]} -------------\n")
        return [1.0 if match else 0.0 for match in matches]
    
    def format_reward(completions, **kwargs):
        import re, os
        # pattern = r"<think>.*?</think>\s*<answer>.*?\[.*?{\"bbox_2d\":\s*\[\s*\d+,\s*\d+,\s*\d+,\s*\d+\s*\]\s*,\s*\"label\":\s*\".*?\"\s*}.*?\].*?</answer>"
        # pattern = r"<think>.*?</think>\s*<answer>\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]\s*</answer>"
        #pattern  = r"<think>.*?</think>\s*<answer>.*?\"bbox_2d\"\s*:\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\].*?</answer>"
        pattern = (
        r"<think>.*?</think>\s*<answer>\s*"
        r"(?:\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]"   # [x1,y1,x2,y2]
        r"|\(\s*\d+\s*,\s*\d+\s*\)\s*,\s*\(\s*\d+\s*,\s*\d+\s*\))"  # (x1,y1),(x2,y2)
        r"\s*</answer>"
    )
    #     pattern = (
    #     r"<answer>\s*"
    #     r"(?:\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]"   # [x1,y1,x2,y2]
    #     r"|\(\s*\d+\s*,\s*\d+\s*\)\s*,\s*\(\s*\d+\s*,\s*\d+\s*\))"  # (x1,y1),(x2,y2)
    #     r"\s*</answer>"
    # )
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- Format reward: {[1.0 if match else 0.0 for match in matches]} -------------\n")
        return [1.0 if match else 0.0 for match in matches]
    
    def ovd_like_format_reward(completions, **kwargs):
        import re, os

        # 单框两种写法
        bbox_brackets = r'\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]'
        bbox_parens   = r'\(\s*\d+\s*,\s*\d+\s*\)\s*,\s*\(\s*\d+\s*,\s*\d+\s*\)'
        bbox_either   = rf'(?:{bbox_brackets}|{bbox_parens})'

        # 多框列表：[[...], [...], ...]，列表中可混用两种写法
        multi_list    = rf'\[\s*(?:{bbox_either})(?:\s*,\s*{bbox_either})*\s*\]'

        # <think> 必须；<answer> 必须；支持单框或多框
        inner_answer  = rf'(?:{bbox_either}|{multi_list})'
        pattern       = rf'<think>.*?</think>\s*<answer>\s*{inner_answer}\s*</answer>'

        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- Format reward: {[1.0 if m else 0.0 for m in matches]} -------------\n")

        return [1.0 if m else 0.0 for m in matches]

        
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        def iou(box1, box2):
            # 标准 IoU（连续几何，不做 ±1）
            x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
            inter_w = max(0, x2 - x1)
            inter_h = max(0, y2 - y1)
            inter = inter_w * inter_h
            area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
            area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
            union = area1 + area2 - inter
            return float(inter) / union if union > 0 else 0.0
        
        def is_valid(b):
            return isinstance(b, (list, tuple)) and len(b) == 4 and (b[2] > b[0]) and (b[3] > b[1])
        def area(box):
            return max(0, box[2]-box[0]) * max(0, box[3]-box[1])
        def is_zero_box(b):
            return isinstance(b, (list, tuple)) and len(b) == 4 and b[0] == b[1] == b[2] == b[3] == 0
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        answer_tag_pattern = r'<answer>(.*?)</answer>'
        # single_img_with_cot = kwargs['single_img_with_cot']
        # image_path = kwargs['image_path'][0][0]
        # is_single_image = 'dataset/coco/train2014/COCO' in image_path
        # if (not single_img_with_cot) and is_single_image:
        #     answer_tag_pattern = r'(.*)'
        #bbox_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        bbox_pattern = (
                        r'\[\s*(?P<x1_a>\d+)\s*,\s*(?P<y1_a>\d+)\s*,\s*(?P<x2_a>\d+)\s*,\s*(?P<y2_a>\d+)\s*\]'
                        r'|\(\s*(?P<x1_b>\d+)\s*,\s*(?P<y1_b>\d+)\)\s*,\s*\(\s*(?P<x2_b>\d+)\s*,\s*(?P<y2_b>\d+)\)'
                    )
        for content, sol in zip(contents, solution):
            reward = 0.0
            pred = None
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    # bbox_match = re.search(bbox_pattern, content_answer)
                    # if bbox_match:
                    #     # bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                    #     pred = [int(bbox_match.group(i)) for i in range(1, 5)]
                    #     gt = sol if isinstance(sol, (list, tuple)) else list(sol)
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        if bbox_match.group("x1_a"):  # [x1,y1,x2,y2]
                            pred = [
                                int(bbox_match.group("x1_a")),
                                int(bbox_match.group("y1_a")),
                                int(bbox_match.group("x2_a")),
                                int(bbox_match.group("y2_a")),
                            ]
                        else:  # (x1,y1),(x2,y2)
                            pred = [
                                int(bbox_match.group("x1_b")),
                                int(bbox_match.group("y1_b")),
                                int(bbox_match.group("x2_b")),
                                int(bbox_match.group("y2_b")),
                            ]

                        gt = sol if isinstance(sol, (list, tuple)) else list(sol)

                        if is_zero_box(gt):
                            reward = 1.0 if is_zero_box(pred) else 0.0
                        else:
                            iou_val = iou(pred, gt) if is_valid(pred) and is_valid(gt) else 0.0
                            if iou_val < 0.3:
                                # 面积相对误差惩罚：area_factor ∈ (0,1]
                                A_gt   = area(gt)
                                A_pred = area(pred) if is_valid(pred) else 0
                                if A_gt > 0:
                                    rel_err = abs(A_pred - A_gt) / float(A_gt)
                                    alpha = 1.0  # 惩罚强度，可调
                                    area_factor = 1.0 / (1.0 + alpha * rel_err)
                                    iou_val *= area_factor

                            reward = iou_val

            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"pred: {pred}\n")
                    f.write(f"Solution: {sol}\n")
        return rewards
    
    @staticmethod
    def presence_reward(completions, solution, **kwargs):
        import re
        import os
        from datetime import datetime
        
        def is_valid(b):
            return isinstance(b, (list, tuple)) and len(b) == 4 and (b[2] > b[0]) and (b[3] > b[1])
        
        def is_zero_box(b):
            return isinstance(b, (list, tuple)) and len(b) == 4 and b[0] == b[1] == b[2] == b[3] == 0
        
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = (
            r'\[\s*(?P<x1_a>\d+)\s*,\s*(?P<y1_a>\d+)\s*,\s*(?P<x2_a>\d+)\s*,\s*(?P<y2_a>\d+)\s*\]'
            r'|\(\s*(?P<x1_b>\d+)\s*,\s*(?P<y1_b>\d+)\)\s*,\s*\(\s*(?P<x2_b>\d+)\s*,\s*(?P<y2_b>\d+)\)'
        )
        
        for content, sol in zip(contents, solution):
            reward = 0.0
            pred = None
            pred_has_diff = None
            gt_has_diff = False
            presence_correct = False
            
            try:
                gt = sol if isinstance(sol, (list, tuple)) else list(sol)
                gt_has_diff = not (gt is None or is_zero_box(gt))
                
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        if bbox_match.group("x1_a"):  # [x1,y1,x2,y2]
                            pred = [
                                int(bbox_match.group("x1_a")),
                                int(bbox_match.group("y1_a")),
                                int(bbox_match.group("x2_a")),
                                int(bbox_match.group("y2_a")),
                            ]
                        else:  # (x1,y1),(x2,y2)
                            pred = [
                                int(bbox_match.group("x1_b")),
                                int(bbox_match.group("y1_b")),
                                int(bbox_match.group("x2_b")),
                                int(bbox_match.group("y2_b")),
                            ]
                
                if pred is None:
                    pred_has_diff = None
                    presence_correct = False
                elif not is_valid(pred) and not is_zero_box(pred):
                    pred_has_diff = None
                    presence_correct = False
                else:
                    pred_has_diff = not (is_zero_box(pred))
                    presence_correct = (gt_has_diff == pred_has_diff)
                
                reward = 1.0 if presence_correct else 0.0
                
            except Exception:
                reward = 0.0
                pass
            
            rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- Presence reward: {rewards} -------------\n")
        
        return rewards
    
    @staticmethod
    def center_area_reward(completions, solution, *, w_dist=0.6, w_area=0.4, eps=1e-6, **kwargs):
        """
        基于中心距离 + 面积相对误差 的加权奖励（越大越好 ∈ [0,1]）。
        - 中心距离分数: 1 - clamp( d / diag_norm, 0, 1 )
        其中 diag_norm 取 max(diag(gt), diag(pred), 1)，保证尺度相对稳定
        - 面积分数:     1 - clamp( |A_pred - A_gt| / (A_gt + eps), 0, 1 )
        （对GT面积的相对误差；若想对称可改分母为 max(A_pred, A_gt) + eps）

        边界规则（与 iou_reward 保持直观一致）：
        - GT 和 Pred 都为 [0,0,0,0] → 1.0
        - GT 为 0 框、Pred 非 0 → 0.0（误检）
        - GT 非 0 框、Pred 为 0 → 0.0（漏检）

        参数:
        w_dist: 中心距离权重（默认 0.6）
        w_area: 面积误差权重（默认 0.4）
        eps   : 数值稳定
        """
        import re, os
        from datetime import datetime

        def is_valid(b):
            return isinstance(b, (list, tuple)) and len(b) == 4 and (b[2] > b[0]) and (b[3] > b[1])

        def is_zero_box(b):
            return isinstance(b, (list, tuple)) and len(b) == 4 and b[0] == b[1] == b[2] == b[3] == 0

        def area(box):
            return max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))

        def center(box):
            return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)

        def diag_len(box):
            w = max(0.0, box[2] - box[0])
            h = max(0.0, box[3] - box[1])
            return (w * w + h * h) ** 0.5

        # 解析 <answer> 里的 bbox（支持 [x1,y1,x2,y2] 或 (x1,y1),(x2,y2)）
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = (
            r'\[\s*(?P<x1_a>\d+)\s*,\s*(?P<y1_a>\d+)\s*,\s*(?P<x2_a>\d+)\s*,\s*(?P<y2_a>\d+)\s*\]'
            r'|\(\s*(?P<x1_b>\d+)\s*,\s*(?P<y1_b>\d+)\)\s*,\s*\(\s*(?P<x2_b>\d+)\s*,\s*(?P<y2_b>\d+)\)'
        )

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        # 归一化权重
        s = max(eps, float(w_dist + w_area))
        w_d = float(w_dist) / s
        w_a = float(w_area) / s

        for content, sol in zip(contents, solution):
            reward = 0.0
            pred = None
            gt = sol if isinstance(sol, (list, tuple)) else list(sol)

            try:
                ans = re.search(answer_tag_pattern, content, re.DOTALL)
                if ans:
                    payload = ans.group(1).strip()
                    m = re.search(bbox_pattern, payload)
                    if m:
                        if m.group("x1_a"):  # [x1,y1,x2,y2]
                            pred = [int(m.group("x1_a")), int(m.group("y1_a")),
                                    int(m.group("x2_a")), int(m.group("y2_a"))]
                        else:                # (x1,y1),(x2,y2)
                            pred = [int(m.group("x1_b")), int(m.group("y1_b")),
                                    int(m.group("x2_b")), int(m.group("y2_b"))]

                # —— 边界处理 —— #
                if is_zero_box(gt) and (pred is None or is_zero_box(pred)):
                    reward = 1.0
                elif is_zero_box(gt) and (pred is not None and not is_zero_box(pred)):
                    reward = 0.0
                elif (not is_zero_box(gt)) and (pred is None or is_zero_box(pred)):
                    reward = 0.0
                else:
                    # 正常计算：中心距离 + 面积误差
                    if not (is_valid(gt) and is_valid(pred)):
                        reward = 0.0
                    else:
                        # 1) 中心距离分数
                        cx_g, cy_g = center(gt)
                        cx_p, cy_p = center(pred)
                        d = ((cx_g - cx_p) ** 2 + (cy_g - cy_p) ** 2) ** 0.5
                        diag_norm = max(diag_len(gt), diag_len(pred), 1.0)
                        dist_score = 1.0 - min(1.0, d / (diag_norm + eps))

                        # 2) 面积分数（对 GT 相对误差）
                        A_gt, A_pred = area(gt), area(pred)
                        rel_err = abs(A_pred - A_gt) / (A_gt + eps)
                        area_score = 1.0 - min(1.0, rel_err)

                        # 3) 加权汇总
                        reward = max(0.0, min(1.0, w_d * dist_score + w_a * area_score))

            except Exception:
                # 解析异常统一给 0 分（也可按需写日志）
                reward = 0.0

            rewards.append(float(reward))

        # if os.getenv("DEBUG_MODE") == "true":
        #     log_path = os.getenv("LOG_PATH")
        #     with open(log_path, "a", encoding="utf-8") as f:
        #         f.write(f"------------- {current_time} CenterArea reward: {reward:.6f} -------------\n")
        #         f.write(f"Content: {content}\n")
        #         f.write(f"pred: {pred}\n")
        #         f.write(f"Solution: {gt}\n")
        return rewards

    @staticmethod
    def DCLR_reward(completions, solution, *, w_dist=0.6, w_area=0.4, iou_threshold=0.3, low_iou_weight=0.3, eps=1e-6, **kwargs):
        """
        合并 IoU 和 center_area_reward 的奖励函数。
        
        设计原则：
        1. IoU 作为主要指标，高 IoU 的得分始终高于低 IoU 的得分
        2. 当 IoU 较低时，通过 center_area_reward 提供更细粒度的反馈
        3. 确保标准化：即使 center_area_reward 很高，低 IoU 的最终得分也不会超过高 IoU
        
        策略：
        - 当 IoU >= iou_threshold 时：主要使用 IoU，center_area_reward 作为小幅调整
        - 当 IoU < iou_threshold 时：使用 IoU + (1 - IoU) * center_area_reward * low_iou_weight
          这样确保最终得分在 [IoU, IoU + (1-IoU)*low_iou_weight] 范围内，不会超过高 IoU 的得分
        
        参数:
        w_dist: center_area_reward 中中心距离权重（默认 0.6）
        w_area: center_area_reward 中面积误差权重（默认 0.4）
        iou_threshold: IoU 阈值，低于此值时使用混合策略（默认 0.3）
        low_iou_weight: 低 IoU 时 center_area_reward 的权重系数（默认 0.3）
        eps: 数值稳定性参数
        """
        import re
        import os
        from datetime import datetime

        def iou(box1, box2):
            """标准 IoU 计算"""
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            inter_w = max(0, x2 - x1)
            inter_h = max(0, y2 - y1)
            inter = inter_w * inter_h
            area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
            area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
            union = area1 + area2 - inter
            return float(inter) / union if union > 0 else 0.0

        def is_valid(b):
            return isinstance(b, (list, tuple)) and len(b) == 4 and (b[2] > b[0]) and (b[3] > b[1])

        def is_zero_box(b):
            return isinstance(b, (list, tuple)) and len(b) == 4 and b[0] == b[1] == b[2] == b[3] == 0

        def area(box):
            return max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))

        def center(box):
            return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)

        def diag_len(box):
            w = max(0.0, box[2] - box[0])
            h = max(0.0, box[3] - box[1])
            return (w * w + h * h) ** 0.5

        def compute_center_area_score(pred, gt, w_dist, w_area, eps):
            """计算 center_area_reward 分数"""
            if not (is_valid(gt) and is_valid(pred)):
                return 0.0
            
            # 1) 中心距离分数
            cx_g, cy_g = center(gt)
            cx_p, cy_p = center(pred)
            d = ((cx_g - cx_p) ** 2 + (cy_g - cy_p) ** 2) ** 0.5
            diag_norm = max(diag_len(gt), diag_len(pred), 1.0)
            dist_score = 1.0 - min(1.0, d / (diag_norm + eps))

            # 2) 面积分数（对 GT 相对误差）
            A_gt, A_pred = area(gt), area(pred)
            rel_err = abs(A_pred - A_gt) / (A_gt + eps)
            area_score = 1.0 - min(1.0, rel_err)

            # 3) 归一化权重
            s = max(eps, float(w_dist + w_area))
            w_d = float(w_dist) / s
            w_a = float(w_area) / s

            # 4) 加权汇总
            return max(0.0, min(1.0, w_d * dist_score + w_a * area_score))

        # 解析模式
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = (
            r'\[\s*(?P<x1_a>\d+)\s*,\s*(?P<y1_a>\d+)\s*,\s*(?P<x2_a>\d+)\s*,\s*(?P<y2_a>\d+)\s*\]'
            r'|\(\s*(?P<x1_b>\d+)\s*,\s*(?P<y1_b>\d+)\)\s*,\s*\(\s*(?P<x2_b>\d+)\s*,\s*(?P<y2_b>\d+)\)'
        )

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        for content, sol in zip(contents, solution):
            reward = 0.0
            pred = None
            gt = sol if isinstance(sol, (list, tuple)) else list(sol)

            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        if bbox_match.group("x1_a"):  # [x1,y1,x2,y2]
                            pred = [
                                int(bbox_match.group("x1_a")),
                                int(bbox_match.group("y1_a")),
                                int(bbox_match.group("x2_a")),
                                int(bbox_match.group("y2_a")),
                            ]
                        else:  # (x1,y1),(x2,y2)
                            pred = [
                                int(bbox_match.group("x1_b")),
                                int(bbox_match.group("y1_b")),
                                int(bbox_match.group("x2_b")),
                                int(bbox_match.group("y2_b")),
                            ]

                # 边界处理：与 iou_reward 保持一致
                if is_zero_box(gt):
                    reward = 1.0 if (pred is not None and is_zero_box(pred)) else 0.0
                elif pred is None or is_zero_box(pred):
                    reward = 0.0
                else:
                    # 计算 IoU
                    iou_val = iou(pred, gt) if is_valid(pred) and is_valid(gt) else 0.0
                    
                    # 计算 center_area_reward
                    center_area_score = compute_center_area_score(pred, gt, w_dist, w_area, eps) if is_valid(pred) and is_valid(gt) else 0.0
                    
                    # 合并策略：确保高 IoU 的得分始终高于低 IoU
                    # if iou_val >= iou_threshold:
                    #     # 高 IoU 时：主要使用 IoU，center_area 作为小幅调整（不超过 5%）
                    #     # reward = IoU + (center_area - IoU) * 0.05，但限制在 [IoU, min(1.0, IoU + 0.05)]
                    #     adjustment = (center_area_score - iou_val) * 0.05
                    #     reward = min(1.0, iou_val + max(0.0, adjustment))
                    # else:
                    #     # 低 IoU 时：使用 IoU + (1 - IoU) * center_area * low_iou_weight
                    #     # 这样确保最终得分在 [IoU, IoU + (1-IoU)*low_iou_weight] 范围内
                    #     # 例如：IoU=0.1, center_area=0.9, low_iou_weight=0.3
                    #     # reward = 0.1 + 0.9 * 0.9 * 0.3 = 0.1 + 0.243 = 0.343
                    #     # 这样即使 center_area 很高，最终得分也不会超过高 IoU 的情况
                    #     reward = iou_val + (1.0 - iou_val) * center_area_score * low_iou_weight
                    #     reward = min(1.0, reward)
                    reward = iou_val + (1.0 - iou_val) * center_area_score

            except Exception:
                reward = 0.0

            rewards.append(float(reward))
            
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} CombinedIoUCenter reward: {reward:.6f} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"pred: {pred}\n")
                    f.write(f"Solution: {sol}\n")
                    if pred is not None and not is_zero_box(gt) and not is_zero_box(pred):
                        iou_val = iou(pred, gt) if is_valid(pred) and is_valid(gt) else 0.0
                        center_area_score = compute_center_area_score(pred, gt, w_dist, w_area, eps) if is_valid(pred) and is_valid(gt) else 0.0
                        f.write(f"IoU: {iou_val:.6f}, CenterArea: {center_area_score:.6f}\n")

        return rewards

    @staticmethod
    def ovd_like_reward(completions, solution, **kwargs):
        """
        Multi-box reward without class labels, aligned with the formula:
            R = min(1, sqrt(N_gt / N_pred)) * mAP(B_pred, B_gt)
        where mAP is computed as the mean precision over IoU thresholds
        t in {0.50, 0.55, ..., 0.95}, after greedy IoU matching.

        Args:
            completions: list of model outputs, each like [{"content": "<answer> ... </answer>"}]
            solution:    list of GT boxes for each sample:
                        - [[x1,y1,x2,y2], ...]  or  []  (no-object)
                        - if a single GT box is given as [x1,y1,x2,y2], it's auto-wrapped
        Kwargs (optional):
            iou_thresholds: list of thresholds, default [0.50..0.95]
        Returns:
            rewards: list[float], one per sample
        """
        import re, os, math
        from datetime import datetime

        # ---------- small helpers ----------
        def iou(box1, box2):
            x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
            iw = max(0, x2 - x1); ih = max(0, y2 - y1)
            inter = iw * ih
            a1 = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
            a2 = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])
            union = a1 + a2 - inter
            return float(inter) / union if union > 0 else 0.0

        def valid(b):
            return isinstance(b, (list, tuple)) and len(b) == 4 and (b[2] > b[0]) and (b[3] > b[1])

        def is_zero_box(b):
            return isinstance(b, (list, tuple)) and len(b) == 4 and b[0]==b[1]==b[2]==b[3]==0

        def to_box_list(gt):
            # Normalize GT to a list of boxes
            if gt is None:
                return []
            if isinstance(gt, dict):
                # best-effort: allow keys commonly seen
                for k in ["bboxes_2d", "bboxes", "boxes", "bbox_list"]:
                    if k in gt: return [bb for bb in gt[k] if valid(bb)]
                if "bbox_2d" in gt and valid(gt["bbox_2d"]): return [gt["bbox_2d"]]
                return []
            if isinstance(gt, (list, tuple)) and len(gt) == 4 and all(isinstance(x, (int, float)) for x in gt):
                return [list(map(int, gt))]
            if isinstance(gt, (list, tuple)) and all(isinstance(x, (list, tuple)) for x in gt):
                out = []
                for bb in gt:
                    if len(bb) == 4 and all(isinstance(v, (int, float)) for v in bb):
                        out.append([int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])])
                return out
            return []

        def parse_all_boxes(text):
            # find all [x1,y1,x2,y2] or (x1,y1),(x2,y2) 遍历所有不重叠的匹配
            pattern = (
                r'\[\s*(?P<x1_a>\d+)\s*,\s*(?P<y1_a>\d+)\s*,\s*(?P<x2_a>\d+)\s*,\s*(?P<y2_a>\d+)\s*\]'
                r'|\(\s*(?P<x1_b>\d+)\s*,\s*(?P<y1_b>\d+)\)\s*,\s*\(\s*(?P<x2_b>\d+)\s*,\s*(?P<y2_b>\d+)\)'
            )
            boxes = []
            for m in re.finditer(pattern, text):
                if m.group("x1_a") is not None:
                    boxes.append([int(m.group("x1_a")), int(m.group("y1_a")),
                                int(m.group("x2_a")), int(m.group("y2_a"))])
                else:
                    boxes.append([int(m.group("x1_b")), int(m.group("y1_b")),
                                int(m.group("x2_b")), int(m.group("y2_b"))])
            # filter invalid & deduplicate exact repeats
            uniq = []
            seen = set()
            for b in boxes:
                if not valid(b): 
                    continue
                t = tuple(b)
                if t not in seen:
                    uniq.append(b); seen.add(t)
            return uniq

        def greedy_iou_match(preds, gts):
            """Return list of matched IoUs (one per matched pair) using greedy max-IoU matching."""
            if not preds or not gts:
                return []
            # Build IoU matrix
            M = [[iou(p, g) for g in gts] for p in preds]
            used_p, used_g = set(), set()
            matches = []
            # Greedy: repeatedly pick the highest IoU remaining
            while True:
                best = (None, None, -1.0)
                for i in range(len(preds)):
                    if i in used_p: continue
                    for j in range(len(gts)):
                        if j in used_g: continue
                        if M[i][j] > best[2]:
                            best = (i, j, M[i][j])
                i, j, v = best
                if i is None or v <= 0.0:
                    break
                used_p.add(i); used_g.add(j)
                matches.append(v)
            return matches

        # ---------- config ----------
        iou_thresholds = kwargs.get(
            "iou_thresholds", [round(0.15 + 0.05*k, 2) for k in range(10)]
        )
        # iou_thresholds = [0.3] # 固定为0.3
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        now_str = datetime.now().strftime("%d-%H-%M-%S-%f")

        for content, sol in zip(contents, solution):
            try:
                # 1) parse predicted boxes
                pred_txt = content
                m = re.search(answer_tag_pattern, content, re.DOTALL)
                if m: pred_txt = m.group(1).strip()
                preds = parse_all_boxes(pred_txt)

                # 2) normalize GT boxes
                gts = to_box_list(sol)

                # remove pure zero boxes if出现
                preds = [b for b in preds if not is_zero_box(b)]
                gts   = [b for b in gts   if not is_zero_box(b)]

                N_pred, N_gt = len(preds), len(gts)

                # 3) empty cases
                if N_gt == 0:
                    reward = 1.0 if N_pred == 0 else 0.0
                    rewards.append(reward)
                    if os.getenv("DEBUG_MODE") == "true":
                        with open(os.getenv("LOG_PATH"), "a", encoding="utf-8") as f:
                            f.write(f"----- {now_str} OVD-like reward (empty GT): {reward}\n")
                            f.write(f"Content: {content}\n")
                            f.write(f"Solution: {sol}\n")
                            f.write(f"Preds: {preds}\nGTs: {gts}\n")
                    continue
                if N_pred == 0:
                    rewards.append(0.0)
                    if os.getenv("DEBUG_MODE") == "true":
                        with open(os.getenv("LOG_PATH"), "a", encoding="utf-8") as f:
                            f.write(f"----- {now_str} OVD-like reward (no preds): 0.0\n")
                            f.write(f"Content: {content}\n")
                            f.write(f"Solution: {sol}\n")
                            f.write(f"Preds: {preds}\nGTs: {gts}\n")
                    continue

                # 4) greedy IoU matching
                matched_ious = greedy_iou_match(preds, gts)

                # # 5) compute mAP surrogate = mean precision@t over thresholds
                # ap_vals = []
                # for t in iou_thresholds:
                #     tp = sum(1 for v in matched_ious if v >= t)
                #     fp = max(0, N_pred - tp)
                #     # fn = max(0, N_gt - tp)  # not needed for precision
                #     precision_t = (tp / float(tp + fp)) if (tp + fp) > 0 else 0.0
                #     ap_vals.append(precision_t)
                # mAP = sum(ap_vals) / len(ap_vals) if ap_vals else 0.0

                # # 6) overlength penalty
                # penalty = min(1.0, math.sqrt(N_gt / float(N_pred))) if N_pred > 0 else 1.0

                # reward = penalty * mAP
                ap_vals = []
                for t in iou_thresholds:
                    tp = sum(1 for v in matched_ious if v >= t)
                    fp = max(0, N_pred - tp)
                    fn = max(0, N_gt  - tp)
                    P  = (tp / float(tp + fp)) if (tp + fp) > 0 else 0.0
                    R  = (tp / float(tp + fn)) if (tp + fn) > 0 else 0.0
                    ap_vals.append(P * R)
                mAP = sum(ap_vals) / len(ap_vals) if ap_vals else 0.0

                # 对称长度惩罚
                penalty = math.sqrt(min(N_gt, N_pred) / float(max(N_gt, N_pred))) if (N_gt > 0 and N_pred > 0) else 0.0

                reward = penalty * mAP
            except Exception:
                reward = 0.0

            rewards.append(reward)

            if os.getenv("DEBUG_MODE") == "true":
                with open(os.getenv("LOG_PATH"), "a", encoding="utf-8") as f:
                    f.write(f"----- {now_str} OVD-like reward: {reward:.6f}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
                    f.write(f"Pred boxes: {preds}\nGT boxes: {gts}\n")
                    # Optionally write mAP/penalty details
        return rewards

    @staticmethod
    def think_quality_reward(completions, **kwargs):
        import re, os, math, collections
        from datetime import datetime
        """
        Evaluate the quality of the <think> content to guide stepwise localization of anomaly regions.
        The score ∈ [0,1] and combines positive signals (comparison, spatial grounding, numeric cues,
        step structure, concision) and negative signals (repetition, vagueness).
        """
        def evaluate_think(think_text):
            score = 0.0
            text = think_text.lower().strip()
            
            # 1) Explicit comparison cues
            comparison_keywords = [
                "compare", "comparison", "difference", "diff", "anomaly", "change",
                "different", "similar", "contrast", "same", "between", "vs", "versus"
            ]
            if any(kw in text for kw in comparison_keywords):
                score += 0.15
            
            # 2) Refers to both images (e.g., left/right, first/second)
            both_refs_patterns = [
                r"\bleft\b.*\bright\b", r"\bright\b.*\bleft\b",
                r"\bfirst\b.*\bsecond\b", r"\bsecond\b.*\bfirst\b",
                r"\bimage\s*1\b.*\bimage\s*2\b", r"\bimage\s*2\b.*\bimage\s*1\b"
            ]
            if any(re.search(p, text) for p in both_refs_patterns):
                score += 0.15
            
            # 3) Spatial grounding (region descriptors)
            region_keywords = [
                "left", "right", "top", "bottom", "center", "upper", "lower", "corner",
                "edge", "middle", "quadrant", "top left", "top right", "bottom left", "bottom right"
            ]
            if any(kw in text for kw in region_keywords):
                score += 0.15
            
            # 4) Difference details (object/attribute oriented)
            diff_keywords = [
                "added", "removed", "missing", "broken", "scratch", "defect", "paint",
                "color", "shape", "texture", "size", "position", "shift", "moved", "crack"
            ]
            if any(kw in text for kw in diff_keywords):
                score += 0.1
            
            # 5) Numeric/coordinate cues toward localization
            #    numbers, coordinate-like tokens, percentages, pixel-like values
            has_number = bool(re.search(r"\b\d{1,4}\b", text))
            has_coord_like = bool(re.search(r"\(\s*\d+\s*,\s*\d+\s*\)", text)) or bool(
                re.search(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]", text)
            )
            if has_number:
                score += 0.1
            if has_coord_like:
                score += 0.15
            
            # 6) Stepwise structure
            step_keywords = ["first", "then", "next", "after", "step", "finally", "second", "third"]
            sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
            if len(sentences) >= 3 or any(kw in text for kw in step_keywords):
                score += 0.15
            
            # 7) Concision vs. verbosity: softly reward adequate length, penalize excessive length
            words = [w for w in re.split(r"\s+", text) if w]
            word_count = len(words)
            if word_count >= 40:
                score += 0.05
            if word_count > 220:
                score -= 0.05
            
            # 8) Repetition penalty: encourage information density
            counter = collections.Counter(words)
            if word_count > 0:
                unique_ratio = len(counter) / float(word_count)
                if unique_ratio < 0.45:
                    score -= 0.1
                elif unique_ratio < 0.55:
                    score -= 0.05
            
            # 9) Vague language penalty
            vague_markers = ["maybe", "perhaps", "seems", "appears to", "it could be", "not sure"]
            if any(vm in text for vm in vague_markers):
                score -= 0.05
            
            # Clamp
            return max(0.0, min(1.0, score))

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        
        think_pattern = r'<think>(.*?)</think>'
        for content in contents:
            reward = 0.0
            think_match = re.search(think_pattern, content, re.DOTALL)
            if think_match:
                think_text = think_match.group(1).strip()
                reward = evaluate_think(think_text)
            
            rewards.append(reward)
            
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- Think quality reward: {rewards} -------------\n")
        
        return rewards

    @staticmethod
    def cls_choice_accuracy_reward(completions, solution, **kwargs):
        import re
        import os
        from datetime import datetime

        # 与 format_reward_base 保持一致：先抽取 content 列表
        completion_contents = [c[0]["content"] for c in completions]

        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        for content, sol in zip(completion_contents, solution):
            # 关键：确保参与正则/strip 的对象是 str/bytes，避免 TypeError
            content_str = content if isinstance(content, (str, bytes)) else str(content)
            sol_str = sol if isinstance(sol, (str, bytes)) else str(sol)

            r = 0.0
            m = re.search(r'<answer>(.*?)</answer>', content_str, flags=re.DOTALL)
            student = m.group(1).strip() if m else content_str.strip()

            def _norm(x: str) -> str:
                return x.strip().replace(' ', '').replace('_', '').replace('.', '').replace('\n', '').lower()

            gt = _norm(sol_str)
            student_n = _norm(student)

            if student_n == gt:
                r = 1.0

            # 非法选项直接强惩罚（仅允许 a/b/c/d）
            if student_n not in ['a', 'b', 'c', 'd']:
                r = -1.0

            rewards.append(r)

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {r} -------------\n")
                    f.write(f"Content(type={type(content)}): {content_str}\n")
                    # f.write(f"Solution(type={type(sol)}): {str(sol)}\n")
                    f.write(f"student_n(type={type(student_n)}): {student_n}  ; gt(type={type(gt)}): {gt}\n")

        return rewards
    
    @staticmethod
    def answer_reward_TLTA(completions, solution, **kwargs):
        import re
        import os
        import json
        from datetime import datetime

        # 解析 <answer>...</answer>
        ans_re = re.compile(r"<answer>\s*([A-Da-d])\s*</answer>", flags=re.DOTALL)

        # 统一拿到文本
        try:
            completion_contents = [c[0]["content"] for c in completions]
        except Exception:
            completion_contents = [str(c) for c in completions]

        # solution 既可能是 list，也可能是单个；转为 list 以 zip
        if isinstance(solution, (list, tuple)):
            sols = list(solution)
        else:
            sols = [solution] * len(completion_contents)

        def _norm(s: str) -> str:
            return s.strip().replace(' ', '').replace('_', '').replace('.', '').replace('\n', '').lower()

        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        for content_str, sol in zip(completion_contents, sols):
            content_str = content_str if isinstance(content_str, (str, bytes)) else str(content_str)
            # 解析 GT：支持 JSON/dict/纯字符
            gt_ans = None
            try:
                if isinstance(sol, str):
                    gt_obj = json.loads(sol)
                    gt_ans = str(gt_obj.get("answer", "")).strip()
                elif isinstance(sol, dict):
                    gt_ans = str(sol.get("answer", "")).strip()
                else:
                    gt_ans = str(sol).strip()
            except Exception:
                gt_ans = str(sol).strip()

            # 学生答案提取
            m = ans_re.search(content_str)
            student = m.group(1).strip() if m else ""

            student_n = _norm(student)
            gt_n = _norm(gt_ans)

            r = 0.0
            # 非法选项强惩罚（限定 a/b/c/d）
            if student_n not in ['a', 'b', 'c', 'd'] or len(student_n) == 0:
                r = -1.0
            else:
                # 合法再比较
                if gt_n in ['a', 'b', 'c', 'd'] and student_n == gt_n:
                    r = 1.0
                else:
                    r = 0.0

            rewards.append(r)

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                try:
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {r} -------------\n")
                        f.write(f"student={student_n} ; gt={gt_n}\n")
                except Exception:
                    pass

        return rewards


    @staticmethod
    def format_reward_base(completions, **kwargs):
        import re
        import os

        pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]

        # 关键：把可能的 dict/list/None 转成 str 再送进正则
        matches = [
            re.search(pattern, c if isinstance(c, (str, bytes)) else str(c), re.DOTALL) is not None
            for c in completion_contents
        ]

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- Format reward: {[1.0 if match else 0.0 for match in matches]} -------------\n")

        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def format_reward_TLTA(completions, **kwargs):
        """
        TLTA 结构格式奖励（基于 adlabel）：
        - adlabel==1（有缺陷）：必须含 <think>、<location>、<type>、<answer>
        - adlabel==0（无缺陷）：必须含 <think>、<answer>，且不得包含 <location>/<type>
        返回 [0.0/1.0] 列表
        """
        import re
        import os
        import json

        # 读取 GT（从 kwargs['solution'] 里来，支持 str/json/dict）
        gt = kwargs.get("solution", None)
        adlabel = 1  # 默认按有缺陷更严格（防止缺 GT 时误放宽）
        if gt is not None:
            try:
                if isinstance(gt, str):
                    gt_obj = json.loads(gt)
                elif isinstance(gt, dict):
                    gt_obj = gt
                else:
                    gt_obj = {}
            except Exception:
                gt_obj = {}
            # 规范成 0/1
            val = gt_obj.get("adlabel", gt_obj.get("label", 1))
            try:
                adlabel = int(val)
            except Exception:
                adlabel = 1
            adlabel = 1 if adlabel not in (0, 1) else adlabel

        require_loc_typ = (adlabel == 1)

        # 独立检测各标签出现与否（不强制顺序，若需强制可再加顺序正则）
        re_think = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
        re_ans   = re.compile(r"<answer>.*?</answer>", re.IGNORECASE | re.DOTALL)
        re_loc   = re.compile(r"<location>.*?</location>", re.IGNORECASE | re.DOTALL)
        re_type  = re.compile(r"<type>.*?</type>", re.IGNORECASE | re.DOTALL)

        try:
            completion_contents = [c[0]["content"] for c in completions]
        except Exception:
            completion_contents = [str(c) for c in completions]

        scores = []
        for c in completion_contents:
            s = c if isinstance(c, (str, bytes)) else str(c)
            has_think = bool(re_think.search(s))
            has_ans   = bool(re_ans.search(s))
            has_loc   = bool(re_loc.search(s))
            has_type  = bool(re_type.search(s))

            if require_loc_typ:
                ok = has_think and has_ans and has_loc and has_type
            else:
                ok = has_think and has_ans and (not has_loc) and (not has_type)

            scores.append(1.0 if ok else 0.0)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- Format reward (adlabel={adlabel}): {scores} -------------\n")
            except Exception:
                pass

        return scores
    
    @staticmethod
    def location_reward_TLTA(completions, solution, **kwargs):
        import re
        import os
        import json

        # --------- 同义词映射（可按需继续扩展）---------
        REGION_SYNONYM = {
            'top left':   ['top left', 'left top', 'upper left', 'left upper', 'top-left', 'upper-left', 'tl'],
            'top':        ['top', 'top center', 'summit', 'upper', 'upper center'],
            'top right':  ['top right', 'right top', 'upper right', 'right upper', 'top-right', 'upper-right', 'tr'],
            'left':       ['left', 'leftward', 'left-hand'],
            'center':     ['center', 'middle', 'centre', 'mid'],
            'right':      ['right', 'rightward', 'right-hand'],
            'bottom left':['bottom left', 'left bottom', 'lower left', 'left lower', 'bottom-left', 'bl'],
            'bottom':     ['bottom', 'bottom center', 'below', 'lower', 'lower center'],
            'bottom right':['bottom right', 'right bottom', 'lower right', 'right lower', 'bottom-right', 'br'],
        }

        # 预构建 归一->标准 键值
        def _norm(s: str) -> str:
            s = str(s).lower()
            s = re.sub(r"[_\-\.]", " ", s)          # -, _, . 统一为空格
            s = re.sub(r"\s+", " ", s).strip()      # 压缩空白
            s = s.rstrip('.')                       # 去句点
            return s

        norm2canon = {}
        for canon, syns in REGION_SYNONYM.items():
            norm2canon[_norm(canon)] = canon
            for si in syns:
                norm2canon[_norm(si)] = canon

        # 提取 <location>...</location>
        loc_re = re.compile(r"<location>\s*([^<]+?)\s*</location>", flags=re.IGNORECASE | re.DOTALL)

        try:
            completion_contents = [c[0]["content"] for c in completions]
        except Exception:
            completion_contents = [str(c) for c in completions]

        if isinstance(solution, (list, tuple)):
            sols = list(solution)
        else:
            sols = [solution] * len(completion_contents)

        scores = []
        for content_str, sol in zip(completion_contents, sols):
            content_str = content_str if isinstance(content_str, (str, bytes)) else str(content_str)

            # 解析 GT（adlabel/location）
            gt_ad = 1
            gt_loc = ""
            try:
                if isinstance(sol, str):
                    gt_obj = json.loads(sol)
                elif isinstance(sol, dict):
                    gt_obj = sol
                else:
                    gt_obj = {}
            except Exception:
                gt_obj = {}
            try:
                gt_ad = int(gt_obj.get("adlabel", 1))
            except Exception:
                gt_ad = 1
            gt_loc = _norm(gt_obj.get("location", ""))

            # 标准化 GT 位置到 canon
            gt_canon = norm2canon.get(gt_loc, gt_loc)

            m = loc_re.search(content_str)
            has_loc = m is not None
            pred_loc_raw = m.group(1) if m else ""
            pred_loc = _norm(pred_loc_raw)
            pred_canon = norm2canon.get(pred_loc, pred_loc)

            if gt_ad == 1:
                # 有缺陷：必须给位置 & 同义等价
                ok = has_loc and (pred_canon == gt_canon) and len(pred_canon) > 0
                scores.append(1.0 if ok else 0.0)
            else:
                # 无缺陷：不应给位置
                ok = not has_loc
                scores.append(1.0 if ok else 0.0)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[location_reward_TLTA] scores={scores}\n")
            except Exception:
                pass

        return scores

    @staticmethod
    def type_reward_TLTA(completions, solution, **kwargs):
        import re
        import os
        import json

        # 轻量类型同义归一，可按需扩展
        TYPE_ALIASES = {
            "scratches": {"scratch", "scratches", "surface scratch", "surface scratches"},
            "defective_painting": {
                "defective painting", "painting defect", "paint defect",
                "coating defect", "paint issue", "paint flaw", "defective_painting"
            },
        }

        def _normalize(s: str) -> str:
            s = str(s).lower()
            s = re.sub(r"[\s_\-\.]+", " ", s).strip()   # 统一空白
            return s

        def to_canonical(s: str) -> str:
            n = _normalize(s)
            # 先 exact 命中
            for canon, alias_set in TYPE_ALIASES.items():
                if n == _normalize(canon) or n in {_normalize(a) for a in alias_set}:
                    return canon
            # 再做极简规约（去空白/下划线/连字符），容忍复数
            n2 = re.sub(r"[\s_\-]", "", n)
            # 简单复数规约
            if n2.endswith("es"):
                n2s = n2[:-2]
            elif n2.endswith("s"):
                n2s = n2[:-1]
            else:
                n2s = n2
            # 与 canon 做同样规约比较
            for canon in TYPE_ALIASES.keys():
                c2 = re.sub(r"[\s_\-]", "", canon.lower())
                if n2 == c2 or n2s == c2:
                    return canon
            return n  # 找不到就返回规约文本

        # 提取 <type>...</type>
        typ_re = re.compile(r"<type>\s*([^<]+?)\s*</type>", flags=re.IGNORECASE | re.DOTALL)

        try:
            completion_contents = [c[0]["content"] for c in completions]
        except Exception:
            completion_contents = [str(c) for c in completions]

        if isinstance(solution, (list, tuple)):
            sols = list(solution)
        else:
            sols = [solution] * len(completion_contents)

        scores = []
        for content_str, sol in zip(completion_contents, sols):
            content_str = content_str if isinstance(content_str, (str, bytes)) else str(content_str)

            # 解析 GT（adlabel/type）
            gt_ad = 1
            gt_type_raw = ""
            try:
                if isinstance(sol, str):
                    gt_obj = json.loads(sol)
                elif isinstance(sol, dict):
                    gt_obj = sol
                else:
                    gt_obj = {}
            except Exception:
                gt_obj = {}
            try:
                gt_ad = int(gt_obj.get("adlabel", 1))
            except Exception:
                gt_ad = 1
            gt_type_raw = gt_obj.get("type", "")

            gt_canon = to_canonical(gt_type_raw)

            m = typ_re.search(content_str)
            has_typ = m is not None
            pred_typ_raw = m.group(1) if m else ""
            pred_canon = to_canonical(pred_typ_raw)

            if gt_ad == 1:
                ok = has_typ and (pred_canon == gt_canon) and len(pred_canon) > 0
                scores.append(1.0 if ok else 0.0)
            else:
                ok = not has_typ
                scores.append(1.0 if ok else 0.0)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[type_reward_TLTA] scores={scores}\n")
            except Exception:
                pass

        return scores

    @staticmethod
    def seg_iou_reward(completions, solution=None, **kwargs):
        import re, ast, os
        from datetime import datetime
        """
        简洁版:
        - seg_gt 为 None(正常)：<seg> None </seg> => 1.0，否则 0.0
        - seg_gt 为 [x1,y1,x2,y2](异常)：奖励 = IoU(<seg>里的框, seg_gt)；格式不对/退化 => 0.0
        不在此处判定 <answer>，分类正确性交给 accuracy 奖励函数。
        """
        contents = [c[0]["content"] for c in completions]

        gts =  kwargs.get("seg_gt", None)
        n = len(contents)

        # 归一化成逐样本列表：None、[4]、或[[4],...]
        if gts is None:
            gt_list = [None] * n
        elif isinstance(gts, (list, tuple)) and len(gts) == 4:
            gt_list = [list(map(float, gts))] * n
        elif isinstance(gts, (list, tuple)) and len(gts) == n:
            gt_list = [
                (list(map(float, g)) if isinstance(g, (list, tuple)) and len(g) == 4 else None)
                for g in gts
            ]
        else:
            # 不匹配的形状，一律当作无 GT
            gt_list = [None] * n

        def parse_seg(text):
            """返回 ('none') 或 (x1,y1,x2,y2) 或 None(非法/缺失)。"""
            m = re.search(r"<seg>\s*(.*?)\s*</seg>", text, flags=re.DOTALL | re.IGNORECASE)
            if not m:
                return None
            inner = m.group(1).strip()
            if inner.lower() == "none":
                return "none"
            # 异常时，必须是 [] 且长度为4
            if not (inner.startswith('[') and inner.endswith(']')):
                return None
            try:
                arr = ast.literal_eval(inner)
            except Exception:
                return None
            if not (isinstance(arr, list) and len(arr) == 4 and all(isinstance(v, (int, float)) for v in arr)):
                return None
            x1, y1, x2, y2 = map(float, arr)
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                return None
            return (x1, y1, x2, y2)

        def iou(a, b):
            if a is None or b is None: return 0.0
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0: return 0.0
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            union = area_a + area_b - inter
            return 0.0 if union <= 0 else float(inter / union)

        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for text, gt in zip(contents, gt_list):
            pred = parse_seg(text if isinstance(text, (str, bytes)) else str(text))

            if gt is None:
                # 正常样本：只有 <seg> None 才给 1.0
                r = 1.0 if pred == "none" else 0.0
            else:
                # 异常样本：计算 IoU
                pbox = pred if isinstance(pred, tuple) else None
                gbox = tuple(map(float, gt)) if gt is not None else None
                r = iou(pbox, gbox)
            rewards.append(r)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Seg_iou reward: {r} -------------\n")
                    f.write(f"pred_seg: {pred} ; gt_seg : {gt}\n")

        return rewards

    
    @staticmethod
    def cosine_length_reward(completions, solution, **kwargs):
        """
        余弦长度奖励（基于选择题准确性）：
        - acc=-1.0（非法选项） → 直接 -1.0
        - acc= 1.0（正确）     → 越短奖励越高（默认 ~0.5→1.0）
        - acc= 0.0（错误）     → 越长惩罚越重（默认 ~-0.5→0.0）

        需要 kwargs['tokenizer']：与生成模型同源的 tokenizer。
        """
        import re, os, math
        from datetime import datetime

        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is None:
            raise ValueError("cosine_length_reward 需要 tokenizer，请通过 kwargs['tokenizer'] 传入。")

        # 读取超参（如未提供则使用默认）
        min_wrong = kwargs.get("cosine_min_len_value_wrong", -0.5)
        max_wrong = kwargs.get("cosine_max_len_value_wrong",  0.0)
        min_corr  = kwargs.get("cosine_min_len_value_correct", 1.0)
        max_corr  = kwargs.get("cosine_max_len_value_correct", 0.5)
        max_len   = kwargs.get("cosine_max_len", 512)
        soft_len  = kwargs.get("soft_cache_length", 256)

        # 提取内容，与其他奖励保持一致
        completion_contents = [c[0]["content"] for c in completions]

        def _norm(x: str) -> str:
            return x.strip().replace(' ', '').replace('_', '').replace('.', '').replace('\n', '').lower()

        def _cosfn(t, T, vmin, vmax):
            # t=0 → vmax；t=T → vmin
            return vmax - (vmax - vmin) * (1 - math.cos(t * math.pi / T)) / 2

        T = max(1, max(0, max_len - soft_len))  # 避免除 0
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        for content, sol in zip(completion_contents, solution):
            content_str = content if isinstance(content, (str, bytes)) else str(content)
            sol_str = sol if isinstance(sol, (str, bytes)) else str(sol)

            # 1) 选择题准确性打分（与 cls_choice_accuracy_reward 一致，但不写日志）
            m = re.search(r'<answer>(.*?)</answer>', content_str, flags=re.DOTALL)
            student = m.group(1).strip() if m else content_str.strip()
            gt = _norm(sol_str)
            student_n = _norm(student)

            if student_n not in ['a', 'b', 'c', 'd']:
                acc = -1.0
            elif student_n == gt:
                acc = 1.0
            else:
                acc = 0.0

            # 2) 基于长度的余弦映射
            if acc == -1.0:
                reward = -1.0
            else:
                gen_len = len(tokenizer.encode(content_str))
                t = max(0, gen_len - soft_len)

                if acc == 1.0:
                    # 正确：越短越高（交换区间两端实现）
                    vmin, vmax = max_corr, min_corr
                else:
                    # 错误：越长越罚
                    vmin, vmax = max_wrong, min_wrong

                reward = _cosfn(t, T, vmin, vmax)

            rewards.append(reward)

            # DEBUG 日志（可通过 DEBUG_MODE/LOG_PATH 控制）
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH", "./debug_log.txt")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} Cosine reward: {reward:.6f} -------------\n")
        return rewards
