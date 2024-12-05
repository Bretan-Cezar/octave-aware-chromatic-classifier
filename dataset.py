from torch.utils.data import IterableDataset
import random
import h5py as h5
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
from audiolazy import lazy_lpc

class NoteDataset(IterableDataset):
    def __init__(self, config: dict, type: str = "train"):
        super(NoteDataset).__init__()

        # plt.ion()

        # self.__fig = plt.figure()
        # ax = self.__fig.add_subplot(111)

        # ax.set_ylim(0, 3)

        # self.__line, = ax.plot(np.arange(0, 16000), np.zeros((16000)), 'r-')
        # self.__fig.canvas.draw()
        # self.__fig.canvas.flush_events()
        
        if type == "test":
            typeKey = "testSubdirs"
        else:
            typeKey = "trainSubdirs"

        self.__config = config

        self.__subdirs = [os.path.join(self.__config["dbRootDir"], subdir) for subdir in self.__config[typeKey]]

        self.__dbs = [[os.path.join(subdir, db) for db in sorted(os.listdir(subdir))] for subdir in self.__subdirs]

        self.__base_data = []
        
        if "addNoise" in self.__config:
            self.__noisy_data = np.ndarray((), dtype=np.float32)

        for subdir in self.__dbs:

            for db in subdir:

                with h5.File(db, 'r') as f:
                    
                    spectra = f["spectra"][()]
                    pitches = f["pitches"][()]
                    
                    if config["lowCut"]:
                        spectra[:, :config["lowCut"]] = 0

                    for pair in tqdm(zip(spectra, pitches), desc=f"Reading data from DB file {db}..."):
                        self.__base_data.append(pair)
        
                
    def __len__(self):
        return len(self.__base_data)

    def __iter__(self):

        random.shuffle(self.__base_data)

        if "addNoise" in self.__config and self.__config["addNoise"]:

            del self.__noisy_data

            self.__noisy_data = []

            for i, (spectrum, pitch) in enumerate(self.__base_data):

                noise = np.abs(
                    np.random.normal(
                        np.random.uniform(0, 0.1), 
                        np.random.uniform(0.5, 2.0) * 0.006, 
                        spectrum.shape
                    )
                    .astype(np.float32)
                )
                noise[:self.__config["lowCut"]] = 0

                noisy_spec = np.copy(spectrum) + noise

                noisy_spec = noisy_spec * np.random.normal(0.95, 0.05)

                # if i % 100 == 0:
                #     self.__line.set_ydata(noisy_spec)
                #     self.__fig.canvas.draw()
                #     self.__fig.canvas.flush_events()
                #     time.sleep(1)

                self.__noisy_data.append((noisy_spec, pitch))
                
            return iter(self.__noisy_data)
    
        else:

            # for i, (spectrum, pitch) in enumerate(self.__base_data):

                # if i % 50 == 0:
                #     self.__line.set_ydata(spectrum)
                #     self.__fig.canvas.draw()
                #     self.__fig.canvas.flush_events()
                #     time.sleep(1)
            
            return iter(self.__base_data)