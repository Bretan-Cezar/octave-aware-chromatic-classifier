from feature_extractor import FeatureExtractor
import json
import os

if __name__ == "__main__":

    config = {}

    with open("./data_config.json", "r") as conf_file:
        config = json.load(conf_file)
    
    if "dataRootDir" not in config:
        raise Exception("Data root directory not specified in config")
    
    data_root_dir = config["dataRootDir"]

    if ("subdirs" not in config) or type(config["subdirs"]) != list:
        raise Exception("List of data subdirectories not specified in config")
    
    subdirs = config["subdirs"]

    if "dbRootDir" not in config:
        raise Exception("DB save root directory not specified in config")
    
    db_root_dir = config["dbRootDir"]

    if "targetSampleRate" not in config:
        raise Exception("Target data sample rate not specified in config")
    
    target_sr = int(config["targetSampleRate"])

    if "samplesPerWindow" not in config:
        raise Exception("No. samples per window not specified in config")
    
    win_samples = int(config["samplesPerWindow"])

    if "samplesPerHop" not in config:
        raise Exception("No. samples per hop not specified in config")
    
    hop_samples = int(config["samplesPerHop"])

    if "windowType" not in config:
        raise Exception("Window type for computing STFT not specified in config")
    
    win_type = config["windowType"]

    if "trimSilence" not in config:
        raise Exception("Whether leading and trailing silence should be trimmed must be specified")
    
    trim = bool(config["trimSilence"])

    if "useDatasetStampsForTrimming" in config:
        use_ds_stamps = bool(config["useDatasetStampsForTrimming"])
    else:
        use_ds_stamps = False
    
    if "highCut" in config:
        high_cut = int(config["highCut"])
    else:
        high_cut = None
    
    extractor = FeatureExtractor(
        data_root_dir=data_root_dir,
        db_root_dir=db_root_dir,
        subdirs=subdirs,
        target_sr=target_sr,
        win_samples=win_samples,
        win_type=win_type,
        hop_samples=hop_samples,
        trim=trim,
        use_ds_stamps=use_ds_stamps,
        high_cut=high_cut
    )

    extractor.extract()