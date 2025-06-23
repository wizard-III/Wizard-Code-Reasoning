#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    :   2025/03/20 13:08:39
@Author  :   wangjiakang 
@File    :   utils.py
'''

import json
import numpy as np
import os
import re
import subprocess
import time
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Tuple, Optional

from rewards import RewardConfig
reward_config = RewardConfig()

from .base import BASE_IMPORTS, BASE_LEETCODE_IMPORTS

_ERROR_MSG_PREFIX = "Failed to execute program: "
_MAX_CHAR_DISPLAY = 2048
CLI_ARG_SIZE_LIMIT = 1024 * 3


def code_exec(code, stdin: str = None, timeout=30):
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env = {k: v for k, v in env.items() if not k.startswith("KML")}

    # Build the firejail command with resource limits and cleanup options
    command = [
        "firejail",
        "--private",
        "--quiet",
        "--seccomp=socket",
        "--profile=pip",
        "--rlimit-nproc=32",
        "--rlimit-nofile=32",
        "--rlimit-fsize=1073741824",  # Limit file size
        "--rlimit-as=4294967296",
        f"--timeout=00:00:{timeout}",
    ]
    command = []

    try:
        with TemporaryDirectory() as tmpdir:
            if len(code) < CLI_ARG_SIZE_LIMIT and "import sys" not in code:
                command.extend(["python", "-c", code])
                result = subprocess.run(command,
                                        cwd=tmpdir,
                                        input=stdin.encode() if stdin else None,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        timeout=timeout,
                                        env=env,
                                        check=False)
            else:
                with NamedTemporaryFile(dir="/tmp", suffix=".py") as tmp:
                    tmp.write(code.encode())
                    tmp.flush()
                    command.extend(["python", tmp.name])
                    result = subprocess.run(command,
                                            cwd=tmpdir,
                                            input=stdin.encode() if stdin else None,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            timeout=timeout,
                                            env=env,
                                            check=False)
        stderr = result.stderr.decode().strip()
        stdout = result.stdout.decode()
        if result.returncode == 0:
            return True, stdout
        return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
    except Exception as e:
        return False, _ERROR_MSG_PREFIX + "subprocess.TimeoutExpired"


def remote_check_stdio(code, stdin, stdout):
    succ, output = code_exec(code=code, stdin=stdin)
    return succ, output, stdin, stdout


def validate_format(processed_str: str):
    pattern = re.compile(r'<think>.*</think>.*```python\n.*```.*', re.DOTALL)
    return bool(pattern.match(processed_str.strip()))


def extract_answer(model_solution):
    pattern = r"```python\n(.*?)```"
    match = re.findall(pattern, model_solution, re.DOTALL)
    if len(match) == 0:
        return None
    else:
        code = match[-1]
        return code


def compute_score(solution_str, ground_truth, data_source='', extro_info=[], is_eval=False, debug=False):
    format_correct = validate_format(solution_str)
    if not format_correct:
        extro_info.append("-" * 16 + "Bad format detected!" + "-" * 16)
        extro_info.append("-" * 16 + "Original Model Output" + "-" * 16)
        extro_info.append(solution_str)
        return reward_config.format_error_reward, "\n".join(extro_info)

    solution_str = solution_str.split("</think>")[1]
    solution_code = extract_answer(solution_str)
    if solution_code is None:
        extro_info.append("-" * 16 + "Bad format detected!" + "-" * 16)
        extro_info.append("-" * 16 + "Original Model Output" + "-" * 16)
        extro_info.append(solution_str)
        return reward_config.format_error_reward, "\n".join(extro_info)

    extro_info.append("-" * 16 + "Extracted Code to Execute" + "-" * 16)

    return grade_answer_code(solution_code, ground_truth, data_source, extro_info, is_eval, debug)


def grade_answer_code(solution_code, ground_truth, data_source, extro_info=[], is_eval=False, debug=False):
    t_start = time.time()

    ground_truth = json.loads(ground_truth)
    if "functional" in ground_truth:
        extro_info.append(solution_code + "\n" + ground_truth["functional"])
    else:
        extro_info.append(solution_code)

    if "functional" in ground_truth:
        solution_code = BASE_IMPORTS + "\n" + BASE_LEETCODE_IMPORTS + "\n" + solution_code
        succ, output = code_exec(solution_code + "\n" + ground_truth["functional"], timeout=60)

        if not succ:
            extro_info.append("!" * 16 + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s" + "!" * 16)
            extro_info.append(output[:_MAX_CHAR_DISPLAY])
            extro_info.append("-" * 16 + "Failed Prompt" + "-" * 16)
            return reward_config.incorrect_reward, "\n".join(extro_info)

    elif "inputs" in ground_truth and "outputs" in ground_truth:
        stdin_list: str = ground_truth["inputs"]
        stdout_list: str = ground_truth["outputs"]

        if not is_eval:
            import random
            max_test_case_num = 16
            max_tests = min(max_test_case_num, len(stdin_list))
            selected_indices = range(len(stdin_list))
            selected_indices = random.sample(selected_indices, max_tests)
            stdin_list = [stdin_list[i] for i in selected_indices]
            stdout_list = [stdout_list[i] for i in selected_indices]

        # Add parallelism
        with ThreadPoolExecutor(max_workers=min(16, len(stdin_list))) as executor:
            futures = [
                executor.submit(remote_check_stdio, solution_code, stdin, stdout)
                for stdin, stdout in zip(stdin_list, stdout_list)
            ]
            for future in as_completed(futures):
                succ, output, stdin, stdout = future.result()
                if isinstance(stdin, list):
                    stdin = ", ".join(map(str, stdin))
                if isinstance(stdout, list):
                    stdout = ", ".join(map(str, stdout))

                if not succ or output.strip() != stdout.strip():
                    output = output[:_MAX_CHAR_DISPLAY]  # truncate output to print
                    extro_info.append("!" * 16 + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s" + "!" * 16)
                    extro_info.append(f"❌Actual: {output if output.startswith(_ERROR_MSG_PREFIX) else repr(output.strip())}")
                    extro_info.append("-" * 16 + "Failed Prompt" + "-" * 16)
                    return reward_config.incorrect_reward, "\n".join(extro_info)
    else:
        raise ValueError(
            f"Current supports for ground-truth are ['functional', 'inputs/outputs'] -- No idea what's: {ground_truth = }"
        )

    extro_info.append("+" * 16 + "Test Execution Passed! (Output)" + "+" * 16)
    extro_info.append(output)

    return reward_config.correct_reward, "\n".join(extro_info)


if __name__ == "__main__":
    ground_truth = json.dumps({
        "functional": '''
def check(candidate):
    assert candidate(nums = [3,1,5,4,2], k = 2) == 4
    assert candidate(nums = [3,1,5,4,2], k = 5) == 5
    assert candidate(nums = [3,2,5,3,1], k = 3) == 4


check(Solution().minOperations)'''
    })

    solution = '''class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        is_added = [False] * k
        count = 0
        n = len(nums)
        for i in range(n - 1, -1, -1):
            if nums[i] > k or is_added[nums[i] - 1]:
                continue
            is_added[nums[i] - 1] = True
            count += 1
            if count == k:
                return n - i
'''   
    solution_str = "<think> I am omniscient. </think> ```python\n{}\n```".format(solution)
    score, extro_info = compute_score(solution_str, ground_truth, debug=True)

    marker = "✅" if score == reward_config.correct_reward else "❌"
    extro_info = marker * 16 + "Reward Calculation" + marker * 16 + "\n" + extro_info + "\n" + marker * 16 + f"Final Rward = {score}" + marker * 16

    print(extro_info + "\n\n")