import os
os.environ["SD_ENABLE_ASIO"] = "1"

import librosa
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import NoteClassifier
from pyfiglet import Figlet
from scipy.signal.windows import hann, blackman
import json

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

STREAM = None
GATE_CLOSED = None
CURRENT_LABEL = None

SILENCE_PEAK_AMP = None
FORMATTER = None
    
def draw_spectrum(fig, line, spectrum):
    
    line.set_ydata(spectrum)
    fig.canvas.draw()
    fig.canvas.flush_events()


def log_fft(buffer, nfft, low_cut=None, high_cut=None):
    
    spectrum = np.log10(1+
        np.abs(
            librosa.stft(
                buffer, 
                n_fft=FFT_BUFFER_SIZE, 
                window=blackman, 
                center=False, 
                win_length=FFT_BUFFER_SIZE
            )
        )
    )

    spectrum = spectrum.reshape((nfft//2+1, ))

    if low_cut != None:
        spectrum[:low_cut] = 0
    
    if high_cut != None:
        spectrum = spectrum[:high_cut]

    return spectrum


def load_model(device, model_path) -> NoteClassifier:
    
    model = NoteClassifier()

    model.load_state_dict(torch.load(model_path, weights_only=True))

    model.eval().to(device)

    return model


if __name__ == "__main__":
    print("Starting program... Ensure silence on audio pipeline")

    config = {}

    with open("./realtime_config.json", "r") as conf_file:
        config = json.load(conf_file)

    device = int(config["device"])

    if "lowCut" not in config:
        config["lowCut"] = None
    else:
        config["lowCut"] = int(config["lowCut"]) if int(config["lowCut"]) > 0 else None

    if device >= 0:
        device = torch.device(f"cuda:{device}")
    else:
        device = torch.device('cpu')

    if device.type == 'cuda':
        torch.cuda.set_device(device)

    FORMATTER = Figlet(font='doom', width=160)

    SR = int(config["sampleRate"])
    ASIO_BUFFER_SIZE = int(config["asioBufferSize"])
    FFT_BUFFER_SIZE = int(config["fftBufferSize"])
    DEVICE_INDEX = int(config["deviceIndex"])
    
    NYQUIST = FFT_BUFFER_SIZE//2 
    CHANNEL = int(config["channel"])

    GATE_LOOK_AHEAD = int(config["gateLookAhead"])
    
    LOW_CUT_LIMIT = int(config["lowCut"])
    HIGH_CUT_LIMIT = int(config["highCut"])
    MODEL_PATH = str(config["modelPath"])

    ANALYSIS = bool(config["analysisMode"])
    INFERENCE = bool(config["inferenceMode"])

    LABEL_TO_PITCH = dict(config["labelToPitch"])

    if ANALYSIS:
        plt.ion()

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_ylim(0, 3)

        line1, = ax.plot(np.arange(0, HIGH_CUT_LIMIT), np.zeros((HIGH_CUT_LIMIT)), 'r-')
        fig.canvas.draw()
        fig.canvas.flush_events()
        
    with torch.no_grad():
            
        if INFERENCE:
            model = load_model(device, MODEL_PATH)

        STREAM = sd.InputStream(
            SR, ASIO_BUFFER_SIZE, DEVICE_INDEX, 1, 'float32', 'low', 
            callback=None, extra_settings=sd.AsioSettings(channel_selectors=[CHANNEL])
        )
        
        STREAM.start()
        
        print("Setting noise floor...")

        silence = STREAM.read(5*SR)[0].reshape((5, SR))

        SILENCE_PEAK_AMP = np.max(silence)

        GATE_CLOSED = True

        print("Done ; Commencing reading...")
        
        gate_buffer = STREAM.read(GATE_LOOK_AHEAD)[0].flatten()

        peak_vol = np.max(gate_buffer)

        while True:

            if peak_vol > SILENCE_PEAK_AMP:
                GATE_CLOSED = False

                buffer = STREAM.read(FFT_BUFFER_SIZE)[0].flatten()

                spectrum = log_fft(buffer, FFT_BUFFER_SIZE, LOW_CUT_LIMIT, HIGH_CUT_LIMIT)

                if ANALYSIS:
                    draw_spectrum(fig, line1, spectrum)

                if INFERENCE:
                    spectrum_tensor = torch.from_numpy(spectrum).to(device)

                    note = int(torch.argmax(model(spectrum_tensor)).cpu().numpy())

                    if note != CURRENT_LABEL:

                        os.system('cls')
                        print(FORMATTER.renderText(LABEL_TO_PITCH[str(note)]))
                        CURRENT_LABEL = note

            else:
                if not GATE_CLOSED:
                    if ANALYSIS: 
                        draw_spectrum(fig, line1, np.zeros((HIGH_CUT_LIMIT)))

                    GATE_CLOSED = True

            gate_buffer = STREAM.read(GATE_LOOK_AHEAD)[0].flatten()
            peak_vol = np.max(gate_buffer)