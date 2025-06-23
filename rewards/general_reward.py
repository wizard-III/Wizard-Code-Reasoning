#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    :   2025/03/20 18:52:57
@Author  :   wangjiakang 
@File    :   general_reward.py
'''

import re
import json
import string
from typing import List, Union

import multiprocessing
from multiprocessing import Manager

from rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType

from rewards.code_utils import grade_answer_code
from rewards.code_utils import extract_answer as code_extract_answer

from rewards.math_utils.utils import grade_answer_sympy, grade_answer_mathd
from rewards.math_utils.utils import extract_answer as math_extract_answer

from rewards.code_utils.livecodebench import run_test as lcb_run_test

# Reward function constants
THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"


def postprocess_lcb_sample(sample):
    sample = json.loads(sample)
    sample_inputs = [sample['input'] for sample in sample]
    sample_outputs = [sample['output'] for sample in sample]
    
    sample_dict = {
        'inputs': sample_inputs,
        'outputs': sample_outputs,
    }
    
    if sample[0].get("testtype") == "functional":
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get("func_name", None)
        assert fn_name is not None, f"Function name is not found, check if your LCB data is preprocessed correctly: {metadata}"
        # Fill in the blank
        sample_dict['fn_name'] = fn_name
    
    sample = {
        'input_output': json.dumps(sample_dict),
    }
    return sample


def lcb_check_correctness(sample, generation, timeout=150, debug=False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    assert len(sample) >= 1, "Sample must contain at least one test case"
    sample = postprocess_lcb_sample(sample)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()

    def _temp_run(sample, generation, debug, result, metadata_list, timeout):
        res, metadata = lcb_run_test(sample, test=generation, debug=debug, timeout=timeout)
        result.append(res)
        metadata_list.append(metadata)

    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(
        timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5
    )
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    if not result:
        return False

    return all(x == True for x in result[0])


class RewardCodeFn(RewardFn):
    """
    Reward function for evaluating code answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, data_source: string, is_eval: bool, input: RewardInput):
        assert input.problem_type == RewardType.CODE, \
            "Invalid problem type: expected 'CODE', but got '{}'".format(input.problem_type)

        problem = input.problem
        model_response = input.model_response

        # Extract solution.
        if THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            model_solution = model_response
            # return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # todo extract_answer
        model_answer = code_extract_answer(model_solution)
        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        if data_source in ['livecodebench', 'livecodebench_v5']:
            is_correct = lcb_check_correctness(ground_truths, model_answer)
        else:
            is_correct, extro_info = grade_answer_code(model_answer, ground_truths, data_source, is_eval=is_eval)

        if is_correct == self.config.correct_reward:
            return RewardOutput(reward=self.config.correct_reward, is_correct=True)

        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)


class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, data_source: string, is_eval: bool, input: RewardInput):
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        
        # Extract solution.
        if THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            model_solution = model_response
        
        model_answer = math_extract_answer(model_solution)
        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]

        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = math_extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                
        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)


def general_reward_fn(data_source: str, solution_str: str, ground_truth: Union[str, List[str]], extra_info=None, enable_llm=False, is_eval=False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm

    if data_source in ['code', 'livecodebench', 'livecodebench_v5', 'humanevalplus']:
        reward_fn = RewardCodeFn(reward_config)
        problem_type = RewardType.CODE
    elif data_source in ['math', 'aime2024', 'aime2025', 'math500']:
        reward_fn = RewardMathFn(reward_config)
        problem_type = RewardType.MATH
    else:
        raise ValueError(
            f"Current supports for data_source are ['code', 'livecodebench', 'livecodebench_v5', 'math', 'aime2024', 'math500'] -- No idea what's: {data_source = }"
        )

    reward_response = reward_fn(data_source, is_eval, RewardInput(problem=solution_str, problem_type=problem_type, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response.is_correct


if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig)
    input = RewardInput(problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.", problem_type=RewardType.MATH, model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]})
    output = reward(input)
    print(output)