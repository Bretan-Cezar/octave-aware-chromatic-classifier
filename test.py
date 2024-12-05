import torch
from tqdm import tqdm
from torch.utils.data import DataLoader 
import json
import os
from dataset import NoteDataset
from model import NoteClassifier

def test(model, device, test_data_loader, no_test_samples):
    correct_count = 0

    with torch.no_grad():

        for (spectra, pitches) in tqdm(test_data_loader, desc="Running inference on test samples..."):

            spectra = spectra.to(device)
            pitches = pitches.to(device)

            pitches_pred = model(spectra)

            correct_count += (torch.argmax(pitches_pred, dim=1) == pitches).sum().item()
    
    return (correct_count / no_test_samples) * 100


if __name__ == "__main__":

    config = {}

    with open("./test_config.json", "r") as conf_file:
        config = json.load(conf_file)
    
    if "dbRootDir" not in config:
        raise Exception("DB root directory not specified in config")
    
    db_root_dir = config["dbRootDir"]
    
    test_subdirs = None

    if "testSubdirs" in config:
        test_subdirs = config["testSubdirs"]

    if "modelPath" not in config:
        raise Exception("Model path not specified in config")
    
    if not os.path.exists(config["modelPath"]):
        raise Exception("Model path not found")

    model_path = config["modelPath"]

    if "device" not in config:
        raise Exception("Device not specified in config (>= 0 -> cuda ; < 0 -> cpu)")
    
    device = int(config["device"])
    
    if device >= 0:
        device = torch.device(f"cuda:{device}")
    else:
        device = torch.device('cpu')

    if device.type == 'cuda':
        torch.cuda.set_device(device)
        
    ds_test = NoteDataset(config, "test")

    no_test_samples = len(ds_test)

    test_data_loader = DataLoader(
        dataset=ds_test,
        batch_size=2,
        num_workers=0
    )

    print(f"Total test dataset length: {no_test_samples} samples\n")

    model = NoteClassifier()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.eval().to(device)

    acc = test(model, device, test_data_loader, no_test_samples)

    print(f"TEST ACC: {'%.2f'%acc}%")