#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    :   2025/06/17 19:21:12
@Author  :   wangjiakang
@File    :   wizard.py
'''


from collections import defaultdict

from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from tqdm import tqdm
from typing import Callable, Optional

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


def parallel_compute_score(evaluation_func, response_str, ground_truth, data_sources, extra_info, enable_llm=False, is_eval=False, max_workers=64):
    # with tqdm(total=len(response_str)) as pbar:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluation_func, data_sources[index], response_str[index], ground_truth[index], None, enable_llm, is_eval): index
            for index in range(len(response_str))
        }
        results = {}
        metadata = {}
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()

    return [results[i] for i in range(len(response_str))]


@register("wizard")
class WizardRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def verify(self, data, is_eval):
        """
        verify the batch and save as ``acc`` tensor
        """
        # batched scoring
        prompt_ids = data.batch['prompts']

        response_ids = data.batch['responses']
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get('extra_info', None)

        assert len(sequences_str) == len(ground_truth) == len(data_sources)

        try:
            scores = parallel_compute_score(
                    self.compute_score,
                    sequences_str,
                    ground_truth,
                    data_sources,
                    extra_info=extra_info,
                    enable_llm=False,
                    is_eval=is_eval
                )
            assert len(scores) == len(sequences_str)

        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            scores = [0. for _ in range(len(sequences_str))]

        # data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False, is_eval: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # batched scoring
        prompt_length = data.batch['prompts'].shape[-1]
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        scores = self.verify(data, is_eval)

        for i in range(len(data)):
            reward_extra_info["acc"].append(float(scores[i]))
            reward = scores[i]
            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length[i].item() - 1] = reward

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor