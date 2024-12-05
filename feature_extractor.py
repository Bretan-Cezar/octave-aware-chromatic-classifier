import librosa
import librosa.effects as fx
import h5py as h5
import os
import json
import numpy as np
from scipy.signal.windows import hann, hamming, blackman, bartlett, boxcar
from tqdm import tqdm

class FeatureExtractor:

    def __init__(
            self, 
            data_root_dir: str, 
            db_root_dir: str, 
            subdirs: list, 
            target_sr: int, 
            win_samples: int, 
            win_type: str, 
            hop_samples: int, 
            trim: bool, 
            use_ds_stamps: bool,
            high_cut: int | None = None
        ):

        self.__data_root_dir = data_root_dir
        self.__db_root_dir = db_root_dir
        self.__subdirs = subdirs
        self.__sr = target_sr
        self.__win_length = win_samples
        self.__hop_length = hop_samples
        self.__trim = trim
        self.__use_ds_stamps = use_ds_stamps

        if win_type == "hann":
            self.__window = hann
        elif win_type == "hamming":
            self.__window = hamming
        elif win_type == "blackman":
            self.__window = blackman
        elif win_type == "bartlett":
            self.__window = bartlett
        else:
            self.__window = boxcar

        self.__high_cut = high_cut


    def extract(self):
        for subdir in self.__subdirs:

            data_subdir_path = os.path.join(self.__data_root_dir, subdir)
            metadata_path = os.path.join(data_subdir_path, "metadata.json")

            db_subdir_path = os.path.join(self.__db_root_dir, subdir)

            if not os.path.exists(db_subdir_path):
                os.makedirs(db_subdir_path)

            files = {}

            with open(metadata_path, "r") as f:
                files = json.load(f)
            
            spectra = []
            pitches = []

            for file in tqdm(files, desc=f"Computing Spectrums for files subdirectory {subdir}..."):
                wave, _ = librosa.load(os.path.join(data_subdir_path, file["filename"]), sr=self.__sr)

                if self.__trim:

                    if (self.__use_ds_stamps) and ("onset" in file) and (file["onset"] != None) and ("offset" in file) and (file["offset"] != None):
                        start = int(file["onset"] * self.__sr)
                        end = min(int(file["offset"] * self.__sr), len(wave))

                        wave = wave[start:end]

                    else:
                        wave, _ = fx.trim(wave)
                
                stft = self.log_fft(wave)

                if self.__high_cut:
                    stft = stft[:self.__high_cut, :]

                for i in range(stft.shape[1]):
                    spectra.append(stft[:, i])
                    pitches.append(file["pitch"])
                

            with h5.File(os.path.join(db_subdir_path, f"{subdir}.h5"), "w") as f:
                f.create_dataset("spectra", data=spectra)
                f.create_dataset("pitches", data=pitches)
        
                
    def log_fft(self, frame):
        
        return np.log10(1+
            np.abs(
                librosa.stft(
                    frame, 
                    n_fft=self.__win_length, 
                    window=self.__window, 
                    center=False, 
                    win_length=self.__win_length, 
                    hop_length=self.__hop_length
                )
            )
        )


