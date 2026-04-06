# analysis/io_utils.py
import os
import pandas as pd


def load_sim_data(base_dir=None, filename="sim_data.csv") -> pd.DataFrame:
    """
    Load the main simulation CSV log produced by CSVLogger.

    Parameters
    ----------
    base_dir : str or None
        Directory containing the 'output' directory.
        If None, assumes this file is in 'analysis/' under base_dir.
    filename : str
        CSV filename inside output/.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns: time, id, x, y, vx, vy, speed
    """
    if base_dir is None:
        # assume this file is in your_project/analysis/
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    output_dir = os.path.join(base_dir, "output")
    path = os.path.join(output_dir, filename)

    df = pd.read_csv(path)
    return df