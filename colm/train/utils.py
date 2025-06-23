import random

import numpy as np
import torch
from trak.projectors import BasicProjector, CudaProjector
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch_dicts):
    if isinstance(batch_dicts, dict):
        batch_dicts = [batch_dicts]

    collated = {}
    keys = batch_dicts[0].keys()

    for key in keys:
        values = [
            example[key][0] if isinstance(example[key], list) and len(
                example[key]) == 1 else example[key]
            for example in batch_dicts
        ]

        if isinstance(values[0], torch.Tensor):
            # Normalize to 1D for pad_sequence if needed
            values = [v.squeeze(0) if v.ndim == 2 and v.shape[0]
                      == 1 else v for v in values]

            # Correct padding value for labels
            padding_value = -100 if key == "labels" else 50295

            collated[key] = pad_sequence(
                values,
                batch_first=True,
                padding_value=padding_value
            )
        else:
            collated[key] = values

    return collated


def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        num_sms = torch.cuda.get_device_properties(
            device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(
            8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector


def shuffle_two_lists_together(list1, list2):
    # Ensure both lists have the same length
    assert len(list1) == len(list2), "Both lists must have the same length"

    # Combine the two lists into a list of tuples
    combined = list(zip(list1, list2))

    # Shuffle the combined list
    random.shuffle(combined)

    # Unzip the combined list back into two lists
    list1_shuffled, list2_shuffled = zip(*combined)

    return list(list1_shuffled), list(list2_shuffled)


def convert_to_ordered_range(arr):
    # Find the unique values and sort them
    unique_values = np.unique(arr)

    # Create a mapping from each unique value to its new value
    value_to_new_value = {val: idx for idx, val in enumerate(unique_values)}

    # Replace each element in the arr with its new value
    new_arr = np.array([value_to_new_value[val] for val in arr])

    return new_arr


def adjust_array_to_threshold(arr, upper_bound, threshold, favor="smallest", large_value=128):
    # Ensure no elements in the arr are 0
    arr[arr == 0] = 1

    # Calculate the initial sum of the arr
    current_sum = np.sum(arr)

    # If the sum is smaller than the threshold, increment the smallest/largest elements
    while current_sum < threshold:
        # Find the index of the smallest/largest element
        if favor == "smallest":
            inc_index = np.argmin(arr)
        else:
            inc_index = np.argmax(arr)
        # Increment the smallest/largest element by 1, but do not exceed the upper bound
        if arr[inc_index] < upper_bound[inc_index]:
            arr[inc_index] += 1
            # Update the sum
            current_sum += 1
        else:
            # If the smallest element is at the upper bound, find the next smallest element
            # Temporarily mark this element as processed
            arr[inc_index] = large_value
            continue

    # Restore any elements temporarily marked as processed
    arr[arr == large_value] = upper_bound[arr == large_value]

    # If the sum is larger than the threshold, decrement the largest elements
    while current_sum > threshold:
        # Find the index of the largest element
        max_index = np.argmax(arr)
        # Decrement the largest element by 1
        arr[max_index] -= 1
        # Update the sum
        current_sum -= 1

    return arr


def increase_array_to_threshold(arr, threshold):
    # TODO: Add upperbound version as adjust_array_to_threshold
    # This edge case doesn't seem to happen in practice
    arr = np.array(arr)
    current_sum = np.sum(arr)
    difference = threshold - current_sum

    if difference < 0:
        raise ValueError(
            "The threshold must be greater than or equal to the current sum of the arr.")

    sorted_indices = np.argsort(arr)

    for i in range(difference):
        arr[sorted_indices[i % len(arr)]] += 1

    return arr


def increase_array_to_threshold_v2(min_arr, max_arr, threshold):
    min_arr = np.array(min_arr)
    max_arr = np.array(max_arr)
    current_sum = np.sum(min_arr)
    difference = threshold - current_sum

    if difference < 0:
        raise ValueError(
            "The threshold must be greater than or equal to the current sum of the arr.")

    diff_indices = np.where(min_arr != max_arr)[0]
    shuffled_indices = np.random.permutation(diff_indices)

    for i in range(difference):
        min_arr[shuffled_indices[i % len(min_arr)]] += 1

    return min_arr


def decrease_array_to_threshold(arr, threshold):
    # TODO: Add upperbound version as adjust_array_to_threshold
    # This edge case doesn't seem to happen in practice
    arr = np.array(arr)
    current_sum = np.sum(arr)
    difference = current_sum - threshold

    if difference < 0:
        raise ValueError(
            "The threshold must be less than or equal to the current sum of the array.")

    # Indices of elements sorted in descending order
    sorted_indices = np.argsort(arr)[::-1]

    for i in range(difference):
        if arr[sorted_indices[i % len(arr)]] > 0:
            arr[sorted_indices[i % len(arr)]] -= 1
        else:
            raise ValueError(
                "Cannot decrease elements further without making them negative.")

    return arr


def get_rank(tensor):
    """
    Get the rank number of each element in a tensor.
    Example:
    [20, 3, 5, 8] -> [3, 0, 1, 2]
    """
    flattened_tensor = tensor.flatten()
    sorted_indices = torch.argsort(flattened_tensor)
    rank_tensor = torch.zeros_like(flattened_tensor).long()
    rank_tensor[sorted_indices] = torch.arange(
        len(flattened_tensor)).to(flattened_tensor.device)

    return rank_tensor
