import torch
'''
Tips：
Provide fixed random noise for MCD and DE
1. I: used for counting
2. NS: Record the size of the corresponding neighboring vehicle (nbr) data based on the batch size
'''

# Copy your I and NS here (From: 1_Fixed_noise_size.py)
I = [0, 1, 2]
NS = [1376, 1016, 222]

n=0
for i in NS:
    # NOISE_LEVEL = [0.328, 0.656, 0.984, 1.312, 1.640, 1.968, 2.297, 2.625, 2.953, 3.281]
    # Replace std (NOISE_LEVEL)
    noise = torch.normal(mean=0, std=3.281, size = [16, i, 2])  # L1 0.1
    # Replace it with your target noise path
    filename = f'/media/NOISE/L_10/noise_{n}.pt'
    torch.save(noise,filename)
    n = n+1
