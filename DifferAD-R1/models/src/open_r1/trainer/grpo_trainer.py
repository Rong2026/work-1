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

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized
import math
import random
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from trl import GRPOTrainer

from accelerate.utils import is_peft_model, set_seed
import PIL.Image

import copy
from torch.utils.data import Sampler
import warnings

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from vlm_modules.vlm_module import VLMBaseModule
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class VLMDAPO_New_Trainer(Trainer):
    """
    Trainer for the Decoupled Advantage Policy Optimization (DAPO) method. This is a variant of GRPO with decoupled clipping
    and additional features like dynamic sampling and overlong filtering.

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        vlm_module: VLMBaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = False,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.vlm_module = vlm_module

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        # FIXME
        # Remember to modify it in the invernvl
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        
        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
            # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)
        model = model_cls.from_pretrained(model_id, **model_init_kwargs)

        # LoRA
        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        if peft_config is not None:
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        # Freeze vision modules
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False

        # # if freeze_llm_modules:
        # if hasattr(model, "model") and hasattr(model.model, "text_model"):
        #     print("Freezing LLM modules (except projection layers)...")
        #     for _, p in model.model.text_model.named_parameters():
        #         p.requires_grad = False

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)  # Zero-3 will split model parameters across different GPUs, cannot directly copy existing model, must initialize
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)  # Generally create a copy, deep copy or freeze parameters
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()
            processing_class = processing_cls.from_pretrained(model_id, trust_remote_code=model_init_kwargs.get("trust_remote_code", None), min_pixels=3136, max_pixels=401408)
            for processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    setattr(processing_class, processing_keyword, kwargs[processing_keyword])
            if getattr(processing_class, "tokenizer",  None) is not None:
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
                pad_token_id = processing_class.pad_token_id

        self.vlm_module.post_model_init(model, processing_class)
        self.vlm_module.post_model_init(self.ref_model, processing_class)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel): 
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token 
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes 

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = None
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1,
            pad_token_id=pad_token_id,
        )
        if hasattr(self.vlm_module, "get_eos_token_id"): # For InternVL
            self.generation_config.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)
            print(222, self.vlm_module.get_eos_token_id(processing_class))
        self.beta = args.beta
        self.epsilon = args.epsilon

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # ===== DAPO flags & hparams =====
        self.epsilon_low  = getattr(args, "epsilon_low",  getattr(args, "epsilon", 0.2))  # DAPO: lower clip
        self.epsilon_high = getattr(args, "epsilon_high", getattr(args, "epsilon", 0.28)) # DAPO: higher clip

        self.dynamic_sampling      = getattr(args, "dynamic_sampling", True)  
        self.overlong_filter       = getattr(args, "overlong_filter", True)  
        self.soft_overlong_punish  = getattr(args, "soft_overlong_punish", False)
        self.length_max            = getattr(args, "length_max", self.max_completion_length or 20480)
        self.length_cache          = getattr(args, "length_cache", 4096)     


        self.ref_model = None

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            try:
                model.gradient_checkpointing_enable()
            except:
                # For InternVL; these operations are copied from the original training script of InternVL
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                # This line is necessary, otherwise the `model.gradient_checkpointing_enable()` will be executed during the training process, leading to an error since InternVL does not support this operation.
                args.gradient_checkpointing = False

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)  # Get log probabilities
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return [e for e in ele]
        else:
            return [ele]

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]], model) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)
        # Handle both pre-loaded images and image paths
        images = []
        for x in inputs:
            if "image" in x: 
                imgs = self._get_key_from_inputs(x, "image")
            elif "image_path" in x and x["image_path"] is not None:
                imgs = [PIL.Image.open(p) for p in self._get_key_from_inputs(x, "image_path")]

            for img in imgs:
                try:
                    # Ensure minimum dimensions of 28 pixels
                    w, h = img.size
                    if w < 28 or h < 28:
                    # Calculate new dimensions maintaining aspect ratio
                        if w < h:
                            new_w = 28
                            new_h = int(h * (28/w))
                        else:
                            new_h = 28
                            new_w = int(w * (28/h))
                        img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                except:
                    pass
                images.append(img)
                

        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]


        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:  # Get the bare model for generation
            generate_returned_result = unwrapped_model.generate(
                **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()}, 
                generation_config=self.generation_config
            ) 
            prompt_length = prompt_ids.size(1)
            if not self.vlm_module.is_embeds_input():
                prompt_completion_ids = generate_returned_result 
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
            else:
                # In this case, the input of the LLM backbone is the embedding of the combination of the image and text prompt
                # So the returned result of the `generate` method only contains the completion ids
                completion_ids = generate_returned_result
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)  # Store the index of the first EOS token in each sample
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]  # Find the position of the first eos_token_id index in each batch, corresponding to (batch_size,)
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        has_eos = is_eos.any(dim=1)
        truncated = ~has_eos 

        # Overlength filtering: truncated samples do not backpropagate gradients (only when overlong_filter=True)
        if self.overlong_filter:
            completion_mask = completion_mask.clone()
            completion_mask[truncated] = 0  # All zeros => this sample does not participate in loss
        # Calculate generation length (including EOS position; if truncated, take maximum length)
        gen_len = torch.where(has_eos, eos_idx + 1, torch.full_like(eos_idx, completion_ids.size(1)))

        # R_length ∈ {0, (Lmax-L - Lcache)/Lcache, -1}
        if self.soft_overlong_punish:
            Lmax, Lcache = self.length_max, self.length_cache
            # piecewise: see DAPO Eq.(13)
            zero = torch.zeros_like(gen_len, dtype=torch.float32)
            neg1 = torch.full_like(gen_len, -1, dtype=torch.float32)
            mid  = ((Lmax - Lcache) - gen_len.float()) / float(Lcache)
            r_len = torch.where(gen_len <= (Lmax - Lcache), zero,
                    torch.where(gen_len <= Lmax, mid, neg1))


        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        # Get the multimodal inputs
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
                old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapters():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask, **multimodal_inputs
                    )
        if ref_per_token_logps is not None:
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        # No need to duplicate prompts as we're not generating multiple completions per prompt

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # No need to duplicate prompts as we're not generating multiple completions per prompt
                        # reward_kwargs[key].extend([example[key]] * self.num_generations)
                        reward_kwargs[key].extend([example[key]])
                reward_kwargs['single_img_with_cot'] = self.args.single_img_with_cot
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        miou = rewards_per_func[:, 0].mean()

        # Added on 12_8
        miou_vec = rewards_per_func[:, 0].detach()   # Shape [B_local]
        # === Threshold-based "error" marking, also [B_local] ===
        thr = getattr(self.args, "iou_correct_thr", 0.3)
        is_incorrect = (miou_vec < thr).float()       # [B_local]

        G = self.num_generations
        incorrect_per_group = is_incorrect.view(-1, G).sum(dim=1)          # [num_groups_local]
        p_err_group = (incorrect_per_group / G)                            # [num_groups_local]

        p_err = p_err_group.repeat_interleave(G, dim=0)                              # [B_local]

        has_anomaly_list = []
        for example in inputs:
            solution = example.get('solution', [0, 0, 0, 0])
            if isinstance(solution, list) and len(solution) >= 4:
                # Check if it's a full zero box (normal sample)
                is_normal = all(abs(x) < 1e-6 for x in solution[:4])
                has_anomaly_list.append(0 if is_normal else 1)
            else:
                # Default to anomaly sample
                has_anomaly_list.append(1)
        has_anomaly = torch.tensor(has_anomaly_list, dtype=torch.float32, device=device)  # [B_local]
        
        # Gather rewards across processes
        rewards_per_func = self.accelerator.gather(rewards_per_func)
        
        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)
        
        # Compute grouped-wise rewards
        # Each group consists of num_generations completions for the same prompt
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        # Get only the local slice of advantages
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        
        if self.dynamic_sampling:
            # print("using dynamic sampling")
            reward_idx = 0

            # Start dynamic resampling process
            max_resample_times = getattr(self.args, "max_resample_times", 3)  # Default maximum resampling 3 times
            resample_count = 0

            # Calculate the data range responsible for the current process
            num_processes = self.accelerator.num_processes
            process_index = self.accelerator.process_index
            local_groups = len(rewards) // self.num_generations  # Number of groups in current process
            total_groups = local_groups * num_processes
            start_group = process_index * local_groups
            end_group = start_group + local_groups

            local_sample_index = None

            while resample_count < max_resample_times:
                # Use the specified reward function for grouping
                gathered_rewards_per_func = self.accelerator.gather(rewards_per_func)
                grouped_rewards = gathered_rewards_per_func[:, reward_idx].view(-1, self.num_generations)

                # Calculate standard deviation for each group (to detect if all answers are the same)
                group_std = grouped_rewards.std(dim=1)

                # Check if reward values are all not equal to 1
                group_all_not_equal_1 = (grouped_rewards != 1.0).all(dim=1)

                # Invalid group condition: all answers are the same (std close to 0) and all reward values are not equal to 1
                group_invalid = (group_std < 1e-6) & group_all_not_equal_1  # True means invalid group
                valid_indices = torch.nonzero(~group_invalid, as_tuple=True)[0]

                # Find invalid groups within current process range
                local_invalid_groups = [idx for idx in range(total_groups) if start_group <= idx < end_group and group_invalid[idx].item()]

                # If current process has invalid groups and has not reached resampling limit, continue resampling
                if local_invalid_groups and resample_count < max_resample_times - 1:
                    # Randomly select some invalid groups for resampling
                    k = min(len(local_invalid_groups), len(inputs) // self.num_generations)
                    local_sample_index = random.sample(local_invalid_groups, k)

                    resample_count += 1
                    if self.is_world_process_zero():
                        print(f"[Reward-based Dynamic Sampling] Resampling {len(local_sample_index)} invalid groups for the {resample_count}th time")

                    # Regenerate groups that need resampling
                    for group_idx_global in local_sample_index:
                        # Calculate the index of this group in the local process
                        local_group_idx = group_idx_global - start_group
                        if 0 <= local_group_idx < len(inputs) // self.num_generations:
                            group_start_local = local_group_idx * self.num_generations
                            group_end_local = group_start_local + self.num_generations

                            if group_end_local <= len(inputs):  # Ensure within local process range
                                # Get the input corresponding to this group
                                group_inputs = inputs[group_start_local:group_end_local]
                                group_prompts = prompts[group_start_local:group_end_local]

                                # Re-prepare the prompt and images for this group
                                group_prompts_text = self.vlm_module.prepare_prompt(self.processing_class, group_inputs)
                                group_images = []
                                for x in group_inputs:
                                    if "image" in x:
                                        imgs = self._get_key_from_inputs(x, "image")
                                    elif "image_path" in x and x["image_path"] is not None:
                                        imgs = [PIL.Image.open(p) for p in self._get_key_from_inputs(x, "image_path")]
                                    else:
                                        imgs = []

                                    for img in imgs:
                                        try:
                                            w, h = img.size
                                            if w < 28 or h < 28:
                                                if w < h:
                                                    new_w = 28
                                                    new_h = int(h * (28/w))
                                                else:
                                                    new_h = 28
                                                    new_w = int(w * (28/h))
                                                img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                                        except:
                                            pass
                                        group_images.append(img)

                                # Re-prepare model inputs
                                group_prompt_inputs = self.vlm_module.prepare_model_inputs(
                                    self.processing_class,
                                    group_prompts_text,
                                    group_images,
                                    return_tensors="pt",
                                    padding=True,
                                    padding_side="left",
                                    add_special_tokens=False,
                                )
                                group_prompt_inputs = super()._prepare_inputs(group_prompt_inputs)

                                # Regenerate completions for this group
                                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                                    group_generate_result = unwrapped_model.generate(
                                        **{k: v for k, v in group_prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()},
                                        generation_config=self.generation_config
                                    )
                                    group_prompt_length = group_prompt_inputs["input_ids"].size(1)
                                    if not self.vlm_module.is_embeds_input():
                                        group_prompt_completion_ids = group_generate_result
                                        group_prompt_ids_new = group_prompt_completion_ids[:, :group_prompt_length]
                                        group_completion_ids_new = group_prompt_completion_ids[:, group_prompt_length:]
                                    else:
                                        group_completion_ids_new = group_generate_result
                                        group_prompt_ids_new = group_prompt_inputs["input_ids"]
                                        group_prompt_completion_ids = torch.cat([group_prompt_ids_new, group_completion_ids_new], dim=1)

                                # Update completion_ids and completion_mask
                                # Ensure the length of newly generated completion_ids matches the original length
                                original_length = completion_ids[group_start_local:group_end_local].size(1)
                                if group_completion_ids_new.size(1) != original_length:
                                    if group_completion_ids_new.size(1) > original_length:
                                        # Truncate to original length
                                        group_completion_ids_new = group_completion_ids_new[:, :original_length]
                                    else:
                                        # Pad to original length using pad_token_id
                                        pad_length = original_length - group_completion_ids_new.size(1)
                                        pad_tensor = torch.full((group_completion_ids_new.size(0), pad_length),
                                                              self.processing_class.pad_token_id,
                                                              dtype=group_completion_ids_new.dtype,
                                                              device=group_completion_ids_new.device)
                                        group_completion_ids_new = torch.cat([group_completion_ids_new, pad_tensor], dim=1)

                                completion_ids[group_start_local:group_end_local] = group_completion_ids_new

                                # Recalculate completion_mask for this group
                                group_is_eos = group_completion_ids_new == self.processing_class.eos_token_id
                                group_eos_idx = torch.full((group_is_eos.size(0),), group_is_eos.size(1), dtype=torch.long, device=device)
                                group_eos_idx[group_is_eos.any(dim=1)] = group_is_eos.int().argmax(dim=1)[group_is_eos.any(dim=1)]
                                group_sequence_indices = torch.arange(group_is_eos.size(1), device=device).expand(group_is_eos.size(0), -1)
                                group_completion_mask_new = (group_sequence_indices <= group_eos_idx.unsqueeze(1)).int()

                                # Apply overlong filtering
                                if self.overlong_filter:
                                    group_has_eos = group_is_eos.any(dim=1)
                                    group_truncated = ~group_has_eos
                                    group_completion_mask_new[group_truncated] = 0

                                completion_mask[group_start_local:group_end_local] = group_completion_mask_new

                                # Re-decode and calculate rewards
                                group_completions_new = self.processing_class.batch_decode(group_completion_ids_new, skip_special_tokens=True)
                                if is_conversational(group_inputs[0]):
                                    group_completions_new = [[{"role": "assistant", "content": c}] for c in group_completions_new]

                                # Recalculate rewards for this group
                                for i, (reward_func, reward_processing_class) in enumerate(
                                    zip(self.reward_funcs, self.reward_processing_classes)
                                ):
                                    if isinstance(reward_func, PreTrainedModel):
                                        if is_conversational(group_inputs[0]):
                                            messages = [{"messages": p + c} for p, c in zip(group_prompts, group_completions_new)]
                                            texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                                        else:
                                            texts = [p + c for p, c in zip(group_prompts, group_completions_new)]
                                        reward_inputs = reward_processing_class(
                                            texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                                        )
                                        reward_inputs = super()._prepare_inputs(reward_inputs)
                                        with torch.inference_mode():
                                            group_rewards = reward_func(**reward_inputs).logits[:, 0]
                                    else:
                                        reward_kwargs = {key: [] for key in group_inputs[0].keys() if key not in ["prompt", "completion"]}
                                        for key in reward_kwargs:
                                            for example in group_inputs:
                                                reward_kwargs[key].extend([example[key]])
                                        reward_kwargs['single_img_with_cot'] = self.args.single_img_with_cot
                                        output_reward_func = reward_func(prompts=group_prompts, completions=group_completions_new, **reward_kwargs)
                                        group_rewards = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

                                    # Update rewards_per_func (local part)
                                    rewards_per_func[group_start_local:group_end_local, i] = group_rewards

                                # Recalculate old_per_token_logps and ref_per_token_logps (if needed)
                                group_attention_mask = torch.cat([group_prompt_inputs["attention_mask"], group_completion_mask_new], dim=1)
                                group_multimodal_inputs = {k: group_prompt_inputs[k] if k in group_prompt_inputs else None for k in multimodal_keywords}

                                with torch.no_grad():
                                    if self.num_iterations > 1:
                                        group_old_logps = self._get_per_token_logps(
                                            model, group_prompt_completion_ids, group_attention_mask, **group_multimodal_inputs
                                        )
                                        old_per_token_logps[group_start_local:group_end_local] = group_old_logps[:, group_prompt_length - 1:]

                                    if self.beta == 0.0:
                                        pass  # ref_per_token_logps is None
                                    elif self.ref_model is not None:
                                        group_ref_logps = self._get_per_token_logps(
                                            self.ref_model, group_prompt_completion_ids, group_attention_mask, **group_multimodal_inputs
                                        )
                                        ref_per_token_logps[group_start_local:group_end_local] = group_ref_logps[:, group_prompt_length - 1:]
                                    else:
                                        with self.accelerator.unwrap_model(model).disable_adapters():
                                            group_ref_logps = self._get_per_token_logps(
                                                model, group_prompt_completion_ids, group_attention_mask, **group_multimodal_inputs
                                            )
                                            ref_per_token_logps[group_start_local:group_end_local] = group_ref_logps[:, group_prompt_length - 1:]
                else:
                    break

            # Recalculate global rewards and advantages (because resampling may have occurred)
            rewards_per_func_updated = self.accelerator.gather(rewards_per_func)
            rewards_updated = rewards_per_func_updated.sum(dim=1)

            # Recalculate grouped rewards and advantages
            mean_grouped_rewards_updated = rewards_updated.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards_updated = rewards_updated.view(-1, self.num_generations).std(dim=1)

            mean_grouped_rewards_updated = mean_grouped_rewards_updated.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards_updated = std_grouped_rewards_updated.repeat_interleave(self.num_generations, dim=0)
            advantages_updated = (rewards_updated - mean_grouped_rewards_updated) / (std_grouped_rewards_updated + 1e-4)

            # Update advantages (only update current process part)
            advantages = advantages_updated[process_slice]

            # Update returned rewards_per_func (for subsequent metrics)
            rewards_per_func = rewards_per_func_updated[process_slice]

            # Update global rewards and std_grouped_rewards (for subsequent metrics)
            rewards = rewards_updated[process_slice]
            std_grouped_rewards = std_grouped_rewards_updated

            if self.is_world_process_zero():
                print(f"[Reward-based Dynamic Sampling] Completed, total resampling {resample_count} times")
        

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "multimodal_inputs": multimodal_inputs,
            "miou": miou,
            "p_err": p_err_group,
            "has_anomaly": has_anomaly,  # Anomaly status flag: [B_local], 1 means anomaly, 0 means normal
        }



    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        # Check if we need to generate new completions or use buffered ones
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1

        # Get the prepared inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        multimodal_inputs = inputs["multimodal_inputs"]
        
        # Concatenate for full sequence
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Get the current policy's log probabilities
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, **multimodal_inputs)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]

        # Get the advantages from inputs
        advantages = inputs["advantages"]

        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        # Compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        # coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        # DAPO: decoupled clip range [1-ε_low, 1+ε_high]
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        # if self.args.miou_adjust:
        #     miou = inputs["miou"]
        #     if self.args.miou_adjust_math == 'power': #Suppress high IoU, want to reduce high IoU weights
        #         per_token_loss = ((1.05 - miou) ** self.args.miou_weight) * per_token_loss
        #     elif self.args.miou_adjust_math == 'exp': #想温和地压低高iou的权重
        #         per_token_loss = (math.exp(1 - miou)) * per_token_loss
        #     elif self.args.miou_adjust_math == 'log': #对低iou惩罚最狠，对低iou比较敏感
        #         per_token_loss = (-math.log(0.02 + miou)) * per_token_loss
        if self.args.difficulty_adjust:
            # print("using difficulty adjust")
            miou = inputs["miou"]
            p_err_group = inputs["p_err"]  # [num_groups_local]
            has_anomaly = inputs["has_anomaly"]  # [B_local]，1表示异常，0表示正常
            # print(f"miou shape: {miou.shape}")
            # print(f"p_err_group shape: {p_err_group.shape}")
            # print(f"has_anomaly shape: {has_anomaly.shape}")
            # print(f"per_token_loss shape: {per_token_loss.shape}")
            
            miou_factor = torch.exp(1 - miou)  # 标量，结果也是标量: torch.Size([])
            # p_err_factor = torch.pow(1 + p_err, 0.5)  # 形状: torch.Size([8])
            # # p_err_factor = p_err_factor.unsqueeze(-1)  # 形状变为 [8, 1]
            # # print(f"p_err_factor shape: {p_err_factor.shape}")  
            # # per_token_loss = miou_factor * per_token_loss  # original
            # # per_token_loss = (miou_factor + p_err_factor) / 2 * per_token_loss  # add
            # per_token_loss = (miou_factor * p_err_factor) * per_token_loss  # plus        
            # 将 p_err_group 扩展到 [B_local]，与 has_anomaly 和 per_token_loss 的 batch 维度对齐
            G = self.num_generations
            p_err = p_err_group.repeat_interleave(G, dim=0)  # [B_local]
            p_err_factor = torch.pow(1 + p_err, 0.5)  # [B_local]
            
            # 权重调整因子：异常样本应用完整难度调整，正常样本应用微弱难度感知
            # per_token_loss 形状: [B_local, seq_len]
            # 将 has_anomaly 和 p_err_factor 扩展为 [B_local, 1]，然后广播到 [B_local, seq_len]
            anomaly_mask = has_anomaly.unsqueeze(1)  # [B_local, 1]
            p_err_factor_expanded = p_err_factor.unsqueeze(1)  # [B_local, 1]
            
            # 计算难度调整因子（异常和正常样本都使用）
            difficulty_factor = (miou_factor + p_err_factor_expanded) / 2  # [B_local, 1]
            
            # 权重因子：
            # - 异常样本：使用完整的难度调整因子
            # - 正常样本：使用微弱的难度感知（乘以系数）
            # if self.is_world_process_zero():
            #     print("anomaly_mask:", anomaly_mask)
            #     print("p_err_factor_expanded:", p_err_factor_expanded)
            #     print("normal_weight_coef:", self.normal_weight_coef)
            # weight_factor = (
            #     anomaly_mask * difficulty_factor +  # 异常样本：完整难度调整
            #     (1 - anomaly_mask) * (1.0 + self.normal_weight_coef * difficulty_factor)  # 正常样本：基础权重 + 微弱难度感知
            # )
            weight_factor = difficulty_factor
            
            # if self.is_world_process_zero():
            #     print("weight_factor:", weight_factor)
            per_token_loss = weight_factor * per_token_loss

        # Add KL penalty if beta > 0
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl

            # Log KL divergence
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Compute final loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / (completion_mask.sum(dim=1) + 1e-8)).mean() #
        # loss = (per_token_loss * completion_mask).sum() / (completion_mask.sum() + 1e-8)

        # Log clip ratio
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self) -> Sampler:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )