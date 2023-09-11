import numpy as np
from scipy.sparse import csr_array, csc_array
from typing import Union
from loguru import logger

# since sklearn BOW is actually a sparse matrix, we either need to manually convert the data, or use csr_array


def read_data(fpath: str, vocab_size: int = 2500) -> csr_array:
    try:
        with open(fpath) as f:
            raw_data = f.readlines()
    except Exception as e:
        logger.error(f"Error {e} happened while attempting loading file")
    try:
        rows = []
        cols = []
        vals = []

        # 1. CSR will have document ids as rows and token ids as colums, a cell value is the token's occurence
        for data in raw_data:
            data_split = data.split()
            rows.append(int(data_split[0]) - 1)
            cols.append(int(data_split[1]) - 1)
            vals.append(int(data_split[2]))

        assert (
            len(rows) == len(cols) == len(vals)
        ), f"Invalid mapping size. Expected rows, cols, and vals to have the same length but found {len(rows)} != {len(cols)} != {len(vals)}"
        csr_mat = csc_array(
            (vals, (rows, cols)),
            shape=(max(rows) + 1, vocab_size),
        ).toarray()

    except Exception as e:
        logger.error(f"Error {e} happened while processing data")
    else:
        logger.info(f"Successfully load csc matrix with shape {csr_mat.shape}")
        logger.info(csr_mat)
        return csr_mat


def read_label(fpath: str) -> np.ndarray:
    with open(fpath) as f:
        raw_data = f.readlines()
    return np.array(raw_data, dtype=np.uint8)
