# Data

1. We provide two test sets used in our experiments, including:
  * Testset
  * Car-following testset
2. Files named with **"traj"** are datasets extracted from the NGSIM data, which include the vehicle IDs surrounding the ego vehicle over a specific period.
3. Files named with **"track"** are dictionary files containing the trajectories of each vehicle and should be used in conjunction with the "traj" files.
4. We recommend using PyCharm software to read and inspect the dataset. In the following example, we provide a method to read the file.

  ```python
  import torch
  import numpy as np

  Testset = torch.load("/home/Data/Testset_traj.pt")
  Testset_track = np.load("/home/Data/Testset_tracks.npy", allow_pickle=True)
  ```
