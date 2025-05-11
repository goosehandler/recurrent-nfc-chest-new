# -*- coding: utf-8 -*-
# @Author: wentao.yu
# @Date:   2023-02-25 16:16:28
# @Last Modified by:   wentao.yu
# @Last Modified time: 2023-02-26 13:49:39

"""
demo1: Plot the NMSE performance as a function of SNR. 

References: 
[1] W. Yu, Y. Shen, H. He, X. Yu, J. Zhang, and K. B. Letaief, “Hybrid far- and near-field channel estimation for THz ultra-massive MIMO via fixed point networks,” 
in Proc. IEEE Global Commun. Conf. (GLOBECOM), Rio de Janeiro, Brazil, Dec. 2022.
[2] W. Yu, Y. Shen, H. He, X. Yu, S. Song, J. Zhang, and K. B. Letaief, “An adaptive and robust deep learning framework for THz ultra-massive MIMO channel estimation,” 
arXiv preprint arXiv:2211.15939, 2022.
"""

import sys
sys.path.append(".") 
sys.path.append("../recurrent-nfc-chest")
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from model import FPN_OAMP
from utils import load_CS_matrix, load_CEdataset, compute_NMSE, load_checkpoint
from dataset import HybridFieldChannelDataset

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print('device = ', device)

# specify the system and network parameters, initialize the network
num_measurements = 512
num_antennas = 1024
A = load_CS_matrix(num_measurements, num_antennas).float()
array_type = 'AoSA'
dirname = './dataset/'
testing_SNR = np.array([i for i in range(0, 21)])
testing_channels = 'THzUMHF_' + array_type + '_testing_channel_300GHz_1024.mat'
channel_path = dirname + testing_channels
testing_NMSE = np.zeros(len(testing_SNR))
lat_layers = 3
contraction_factor = 0.99
eps = 1e-2
max_depth = 15
structure = 'ResNet'
num_channels = 64


data_args = {
    'set_size': 100,
    'miniset_size': 100,
    'n_antennas': 1024,
    'n_rf': 4, 
    'n_paths_min': 5,
    'n_paths_max': 5,
    'carrier_freq': 300e9,
    'los_path_len': 30,
    'scat_dist_min': 10,
    'scat_dist_max': 25,
    # 'snr_db_min': SNR_range[0],
    # 'snr_db_max': SNR_range[1],
    'meas_mat_path': '/home/shmatok/recurrent-nfc-chest/data/CSmatrix1024_512_AoSA.npy',
    'n_workers': 40,
    'n_subcarriers': 32,
    'bandwidth': 15e9
}

net = FPN_OAMP(A=A, lat_layers=lat_layers, contraction_factor=contraction_factor,
                eps=eps, max_depth=max_depth, structure=structure, num_channels=num_channels, device=device).to(device)
# device = net.device()

# load the trained network and then test the performance
for i in range(len(testing_SNR)):
    if testing_SNR[i] < 10:
        checkpoint_PATH = f'./checkpoints_wideband/FPN_OAMP_ResNet_weights_0to10dB.pth'
    else:
        checkpoint_PATH = f'./checkpoints_wideband/FPN_OAMP_ResNet_weights_10to20dB.pth'
    net = load_checkpoint(net, checkpoint_PATH, device)
    net.eval()

    # testing_measurements = 'THzUMHF_' + array_type + '_testing_' + str(num_measurements) + '_measurements_' + str(testing_SNR[i]) + 'dB.mat'
    # measurement_path = dirname + testing_measurements
    # measurements, channels, _ = load_CEdataset(measurement_path, channel_path)  # Change here for dadaset generation
    # measurements = measurements.to(device)
    # channels = channels.to(device)
    data_args['snr_db_min'] = testing_SNR[i]
    data_args['snr_db_max'] = testing_SNR[i]


    dataset = HybridFieldChannelDataset(**data_args, device=device)
    dataset.update_dataset(1000)

    loader = torch.utils.data.DataLoader(dataset, batch_size=50)

    for channels, measurements in tqdm(loader):
        channels = channels.reshape((-1, 2048))
        measurements = measurements.reshape((-1, 1024))
        start_time = time()
        channels_pred = net(measurements)
        print(time() - start_time)
        testing_NMSE[i] += compute_NMSE(channels_pred, channels)
    testing_NMSE[i] /= len(loader)
    dataset.delete_dataset()

    # channels = dataset.channels
    # measurements = dataset.measurements
    # use part of the testing dataset for the demo, to avoid out-of-memory issue
    # measurements = measurements[0:500,:]
    # channels = channels[0:500,:]

    # perform testing

# print the testing result in terms of NMSE
print(testing_NMSE)
np.savetxt(f"model_wideband_validation_5_wideband.txt", np.array([testing_SNR, testing_NMSE]).T)

# demo1: Plot the NMSE performance as a function of SNR
'''plt.switch_backend('agg')

plt.figure(1)
l1 = plt.plot(testing_SNR,testing_NMSE,'ro-',label='Proposed FPN-OAMP')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (dB)')
plt.legend()
plt.grid()
plt.show()

plt.savefig("./figures/demo1_NMSE_vs_SNR.jpg")'''