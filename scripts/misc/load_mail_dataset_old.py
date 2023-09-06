import numpy as np
from typing import Union
from loguru import logger

# This file is archived, since sklearn NB doesn't use 3D ndarray

def _pad_array(
    data_list: list,
    pad_length: int,
    padding_mode: str = "mean",
) -> np.ndarray:
    
    """Function for padding sentence
    Args:
        data_list(list): Input data list for padding, current available for 3D array only
        pad_length(int): The final length of the padded sentence
        padding_mode(str): The mode for padding. Default: mean

    Returns:
        np.ndarray: Numpy ndarray that is padded
    """

    # 0. Initialization:
    #   - Calculate max, min, mean length for pad_length-> store in dict
    max_length = max((len(data) for data in data_list))
    min_length = min((len(data)) for data in data_list)
    padding_mode_vals = dict(
        max=max_length, min=min_length, mean=int((max_length + min_length) / 2)
    )

    #   - Assign pad_length if not provided
    if not pad_length:
        padding_mode = "mean" if not padding_mode else padding_mode
        pad_length = padding_mode_vals[padding_mode]

    logger.info(
        f"Padding mode: {padding_mode} with max sentence length as {pad_length}"
    )

    # 1. Perform padding
    res = []
    #   - pad the inner data of the list:
    #       + if the data list's length is smaller than the require pad length, we add the entire list to the new array and add extra [0,0]s until the list meet the require length
    #       + otherwise, we just have to trim our original list
    #   - append x to res
    for data in data_list:
        x = []
        if len(data) < pad_length:
            x[: len(data)] = data
            x[len(data) :] = [[0, 0]] * (pad_length - len(data))
        else:
            x[:pad_length] = data[:pad_length]

        res.append(x)

    # 2. Convert to np's ndarray and return
    return np.array(res)


def read_data(
    fpath: str,
    max_sentence_length: int = None,
    padding_mode: str = "",
) -> np.ndarray:
    
    """ Read feature files function
    
    Args: 
        fpath(str): the path to the data file
        max_sentence_length(int): the length of each sentence in the document, 
        padding_mode(str): the mode for padding if a sentence is shorter than max_sentence_length.

    Returns:
        np.ndarray: The dataset in np.ndarray format
    """
    # 0. Load the text file dataset, view more way of reading text file here https://www.geeksforgeeks.org/reading-writing-text-files-python/
    with open(fpath) as f:
        raw_data = f.readlines()

    data_list = {}
    # 1. Convert the bag of words format to the format: dataset->[document->[tokens and frequency->[]]]
    for data in raw_data:
        data_split = data.split()
        idx = int(data_split[0])
        if idx not in data_list:
            data_list[idx] = []
        data_list[int(data_split[0])].append([int(data_split[1]), int(data_split[2])])

    data_list = list(data_list.values())

    arr = _pad_array(
        data_list, pad_length=max_sentence_length, padding_mode=padding_mode
    )

    return arr


def read_label(fpath: str) -> np.ndarray:
    with open(fpath) as f:
        raw_data = f.readlines()
    return np.array(raw_data)