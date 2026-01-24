# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from trainer import VLMGRPOTrainer, GRPOConfig
from vlm_modules import *
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
from transformers.utils import logging
import torch
from typing import Tuple

logger = logging.get_logger(__name__)
def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward

GENERAL_QUESTION_PROMPT = (
        'You are an expert in detecting defects in image. I will provide you with two images: a reference image (first) showing a normal object without defects, and a test image (second) that needs inspection.'
        'Your task is to compare these images and determine if there are any differences or anomalies in the test image. Use the reference image as a baseline for what is considered normal.'
        '{Question}'
    )

SYSTEM_PROMPT = """
[System Instruction]

When responding, follow these rules:

1. Enclose your full reasoning strictly inside <think></think>.
2. Enclose ONLY the final bounding-box result inside <answer></answer>.
3. The content inside <answer></answer> must contain ONLY coordinates, formatted as:
   (x1,y1),(x2,y2)
   If no abnormal region exists, output:
   (0,0),(0,0)
4. Do not output any text outside these tags except the required description and analysis.
"""

# ================================
# 20 English Prompts
# ================================
PROMPT_POOL = [

    # 1–10 (first batch)
    "Compare the two images and identify any abnormal regions in the target image, including structural, content, or logical inconsistencies. Describe the target briefly and localize the abnormal areas. Ignore differences caused only by lighting or viewpoint variation.",

    "Analyze the reference and target images and determine whether the target contains any unusual or anomalous regions. These may include missing elements, added content, distortions, or logically inconsistent objects. Light or angle changes should not be treated as anomalies.",

    "Examine the two images and find regions in the target that appear abnormal when contrasted with the reference. Consider both visual and semantic irregularities, but disregard lighting or perspective changes. Provide a concise description of the target.",

    "Inspect both images and locate any regions in the target that appear anomalous in structure, appearance, or semantic meaning. Ignore illumination and camera-angle variations. Summarize the target before listing anomalies.",

    "Determine whether the target image contains anomalies by comparing it with the reference. These anomalies may include structural deviations, unexpected content, or logically inconsistent features. Ignore purely photometric differences.",

    "Identify any abnormal or unexpected regions within the target image relative to the reference. This may include physical changes, foreign objects, distortions, or inconsistencies in scene logic. Exclude lighting changes.",

    "Examine the two images side by side and detect anomalies present in the target image. These may be structural, semantic, or logic-level irregularities. Lighting or viewpoint fluctuations should be ignored.",

    "Compare the reference and target and highlight regions in the target that appear abnormal or inconsistent with expected content. Consider both visible and logical abnormalities, ignoring non-essential imaging variations.",

    "Analyze the target image with the reference as baseline and identify any abnormal areas—whether structural, content-related, or logically inconsistent. Ignore differences arising purely from lighting or minor pose shifts.",

    "Look for anomalies in the target image by contrasting it with the reference, including structural changes, unexpected insertions or deletions, and logic-related inconsistencies. Lighting and viewing-angle changes are not anomalies.",

    # 11–20 (second batch, richer expression)
    "Carefully observe the reference and target images and determine whether the target shows any irregularities. These may involve altered shapes, missing details, unexpected insertions, or inconsistencies in scene logic. Ignore lighting and viewpoint changes as they do not indicate abnormalities. Provide a short description of the target.",

    "Compare the two images with attention to meaningful visual or semantic deviations. Identify any areas in the target image that appear unusual or inconsistent with the reference, while disregarding illumination or camera-angle differences. Summarize what you observe in the target before specifying the abnormal regions.",

    "Examine the image pair and identify regions in the target that stand out as abnormal or unexpected compared with the reference. Abnormalities can include structural distortions, misplaced elements, or logically impossible content. Do not treat brightness or viewpoint variations as anomalies.",

    "Your task is to analyze how the target image differs from the reference in a way that indicates abnormalities. These may be visual defects, unlikely configurations, missing components, or semantic contradictions. Ignore lighting and angle variations when assessing differences.",

    "Inspect the target image relative to the reference and highlight any regions that appear abnormal or inconsistent with normal expectations. Differences solely caused by lighting, shading, or camera pose should be disregarded. Begin with a brief description of the target.",

    "Determine whether the target deviates from the reference in any unusual or abnormal way. This may include unexpected objects, distortions, altered structures, or contradictions in scene semantics. Ignore non-structural changes like illumination or perspective shifts.",

    "Analyze both images and look for parts of the target that appear suspicious, inconsistent, or abnormal compared to the reference. Consider structural, content-level, and logical anomalies. Disregard lighting-related or viewpoint-related variations.",

    "Evaluate the target image using the reference as a baseline and locate any regions that seem abnormal, unexpected, or logically inconsistent. These may reflect structural or semantic deviations. Brightness or angle differences should be ignored.",

    "Observe the two images and identify any parts of the target that do not align with the expected appearance shown in the reference. Such anomalies may involve visual inconsistencies, unnatural changes, or semantic irregularities. Do not treat lighting or viewpoint changes as meaningful differences.",

    "Compare the images and determine whether the target contains any abnormal or surprising variations relative to the reference. These may involve changes in structure, content, or logical coherence. Ignore differences caused solely by illumination or viewing angle. Provide a concise description of the target before localizing anomalies."
]
# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
        trainer_type (`str`):
            Type of trainer to use. Possible values: 'grpo', 'dapo', 'gspo'. Default: 'grpo'.
    """

    # reward_funcs: list[str] = field(
    #     default_factory=lambda: ["accuracy", "format", "cot"],
    #     metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    # )
    reward_funcs: list[str] = field(
        # default_factory=lambda: ["accuracy", "format", "center_area"],
        default_factory=lambda: ["accuracy", "format"],
        # default_factory=lambda: ["accuracy", "format", "presence", "center_area"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'presence', 'center_area'"},
    )
    trainer_type: str = field(
        default="grpo", # default="grpo",
        metadata={"help": "Type of trainer to use. Possible values: 'grpo'."},
    )
    max_pixels: Optional[int] = field(
        default=14*14*4*1280,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    iou_center: bool = field(
        default=True,
        metadata={"help": "Whether to use combined IoU and center reward for accuracy. If True, use combined_iou_center_reward; if False, use iou_reward."},
    )
    max_resample_times: Optional[int] = field(
        default=3,
        metadata={"help": "Maximum number of resampling times for the dataset. Default: 3."},
    )
    sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples to randomly select from the dataset. If None, use all data. Default: None."},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments, question_template: str, single_img_with_cot: bool):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []
        self.question_template = question_template
        self.single_img_with_cot = single_img_with_cot

        # with open(data_path, "r") as json_file:
        #     cur_data_dict = json.load(json_file)
                # 这里修改：读取 jsonl 文件
        with open(data_path, "r") as f:
            for line in f:
                if line.strip():
                    self.list_data_dict.append(json.loads(line))


        if script_args.sample_size is not None and script_args.sample_size < len(self.list_data_dict):
            print(f"Original dataset size: {len(self.list_data_dict)}")
            print(f"Sampling {script_args.sample_size} examples from dataset...")
            random.seed(42)
            self.list_data_dict = random.sample(self.list_data_dict, script_args.sample_size)
            print(f"Sampled dataset size: {len(self.list_data_dict)}")
    
    def ref_to_abs(self, ref):
        if os.path.isabs(ref):
            return os.path.normpath(ref)
        root = getattr(self.script_args, "image_root", None)
        if root :
            return os.path.normpath(os.path.join(root, ref))

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        QUESTION_TEMPLATE = self.question_template
        def make_conversation_image(example):
            return {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            # {"type": "image"},
                            *({'type': 'image', 'text': None} for _ in range(len(example['image_path']))),
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                        ],
                    },
                ],
            }

        example = self.list_data_dict[i]

        image_rel = example.get('images')
        example['image_path'] = [self.ref_to_abs(img) for img in image_rel]
        # example['image_path'] = [self.ref_to_abs(img) for img in image_rel[::-1]] #反序，将要检测的图像放在第一张
        # example['image_path'] = [self.ref_to_abs(image_rel[-1]) ] #单图，仅异常图

        # example['problem'] = example.get('prompt', '').replace('<image>', '').strip()
        # example['problem'] = "Please compare the two given images and determine whether there are any differences or anomalies. Provide a detailed description of the target image, and if anomalies exist, focus on explaining the anomalous regions and identify their approximate locations. Output the localized anomalous regions using bounding boxes, with coordinates formatted as [x1, y1, x2, y2]; if no anomaly exists, output [0, 0, 0, 0]. Enclose your reasoning process within <think></think> tags, and enclose your final bounding box answer within <answer></answer> tags."
        # example['problem'] = "Output the localized anomalous regions using bounding boxes, with coordinates formatted as (x1,y1),(x2,y2); if no anomaly exists, output (0,0),(0,0). Enclose your reasoning process within <think></think> tags, and enclose your final bounding box answer within <answer></answer> tags."
        # example['problem'] = "Please compare the two given images and determine whether there are any differences. Provide a detailed description of the target image, and if differences exist, focus on explaining the different regions and identify their approximate locations. Output the localized different regions using bounding boxes, with coordinates formatted as (x1,y1),(x2,y2); if  the images are exactly the same, output (0,0),(0,0). Enclose your reasoning process within <think></think> tags, and enclose your final bounding box answer within <answer></answer> tags."
        # example['problem'] = "Please compare the two given images and determine whether there are any differences. Provide a detailed description of the target image, and if differences exist, focus on explaining the different regions and identify their approximate locations. Output the localized different regions using bounding boxes, with coordinates formatted as [x1, y1, x2, y2]; if  the images are exactly the same, output [0, 0, 0, 0]. Enclose your final bounding box answer within <answer></answer> tags."
        # example['solution'] = example.get('seg_gt')
        example['problem'] = SYSTEM_PROMPT + random.choice(PROMPT_POOL)
        example['solution'] = example.get('seg_gt_qwen2vl')
        
        return {
            'image_path': example['image_path'],
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': make_conversation_image(example)['prompt'],
        }


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)

    # Load the reward functions
    # Choose accuracy reward function based on iou_center parameter
    accuracy_reward = vlm_module_cls.DCLR_reward if script_args.iou_center else vlm_module_cls.iou_reward
    # accuracy_reward = vlm_module_cls.center_area_reward

    reward_funcs_registry = {
        "accuracy": accuracy_reward,
        "format": vlm_module_cls.format_reward,
        # "presence": vlm_module_cls.presence_reward,
        # "center_area":vlm_module_cls.center_area_reward,
        # "cot": vlm_module_cls.think_quality_reward,
    }
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args, question_template=vlm_module_cls.get_question_template(task_type="no"), single_img_with_cot=training_args.single_img_with_cot)

    # Select trainer based on trainer_type argument
    trainer_type = script_args.trainer_type.lower()
   
    trainer_cls = VLMGRPOTrainer
    
    # Initialize the trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
        max_resample_times=script_args.max_resample_times,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
