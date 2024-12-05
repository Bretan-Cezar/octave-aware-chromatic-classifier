import os
import json
import time
import numpy as np
from model import NoteClassifier
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.adam import Adam
from torch.nn import CrossEntropyLoss
from dataset import NoteDataset
from torch.utils.data import DataLoader
import shutil
from tqdm import tqdm
from datetime import datetime
from test import test

def train(model, optimizer, criterion, train_data_loader, test_data_loader, no_train_samples, no_test_samples, config):
    stats = SummaryWriter(os.path.dirname(config["logDir"]))

    max_epochs = config["maxEpochs"]
    ckpt_epochs = config["ckptEpochs"]
    model_dir = config["modelDir"]

    print("TRAINING STARTED".center(shutil.get_terminal_size().columns, "="))
    
    max_test_acc = 0.

    for epoch in range(1, max_epochs+1):

        epoch_loss = 0.
        epoch_acc = 0.

        for (spectra, pitches) in tqdm(train_data_loader, desc=f"Epoch {str(epoch).zfill(4)} / {max_epochs}"):

            optimizer.zero_grad()

            spectra = spectra.to(device)
            pitches = pitches.to(device)

            pitches_pred = model(spectra)

            loss = criterion(pitches_pred, pitches)

            loss.backward()

            optimizer.step()

            epoch_acc += (torch.argmax(pitches_pred.detach(), dim=1) == pitches).sum().item()
            epoch_loss += loss.item() * spectra.shape[0]
        
        epoch_loss /= no_train_samples
        epoch_acc = (epoch_acc / no_train_samples) * 100

        print(f"LOSS: {epoch_loss} ; TRAIN ACC: {'%.2f'%(epoch_acc)}%")
        stats.add_scalar("TRAIN ACC", epoch_acc, epoch)
        stats.add_scalar("TRAIN LOSS", epoch_loss, epoch)

        if epoch % ckpt_epochs == 0:
            test_acc = test(model, device, test_data_loader, no_test_samples)
            
            stats.add_scalar("TEST ACC", test_acc, epoch)

            print(f"TEST ACC at epoch {epoch}: {'%.2f'%(test_acc)}%")
            
            if test_acc > max_test_acc:

                max_test_acc = test_acc

                print("New test acc best, saving model...")

                torch.save(
                    model.state_dict(), 
                    os.path.join(
                        model_dir, 
                        f"ckpt-{epoch}-{'%.2f'%(epoch_acc)}-{'%.2f'%(test_acc)}-{datetime.now().replace(microsecond=0).isoformat().replace(':', '.')}.pt"
                    )
                )


if __name__ == "__main__":

    config = {}

    with open("./train_config.json", "r") as conf_file:
        config = json.load(conf_file)
    
    if "dbRootDir" not in config:
        raise Exception("DB root directory not specified in config")
    
    db_root_dir = config["dbRootDir"]

    if ("trainSubdirs" not in config) or type(config["trainSubdirs"]) != list:
        raise Exception("List of data subdirectories not specified in config")
    
    train_subdirs = config["trainSubdirs"]
    
    test_subdirs = None

    if "testSubdirs" in config:
        test_subdirs = config["testSubdirs"]

    if "modelDir" not in config:
        raise Exception("Model save directory not specified in config")
    
    if not os.path.exists(config["modelDir"]):
        print(f"Created checkpoints directory {config['modelDir']}")
        os.makedirs(config["modelDir"])

    if "logDir" not in config:
        config["logDir"] = ".\\log"
        
    if not os.path.exists(config["logDir"]):
        os.makedirs(config["logDir"])   
    
    if "learningRate" not in config:
        raise Exception("Learning rate not specified in config")
    
    learning_rate = float(config["learningRate"])
    
    if ("beta1" not in config) or ("beta2" not in config) or ("eps" not in config):
        raise Exception("Adam optimizer params (eps, beta1, beta2) must be specified in config")
    
    beta1 = float(config["beta1"])
    beta2 = float(config["beta2"])
    eps = float(config["eps"])

    if "batchSize" not in config:
        raise Exception("Batch size not specified in config")
    
    batch_size = int(config["batchSize"])

    if "maxEpochs" not in config:
        raise Exception("Max no. training epochs not specified in config")
    
    config["maxEpochs"] = int(config["maxEpochs"])
    
    if "ckptEpochs" not in config:
        raise Exception("No. epochs for checkpointing not specified in config")
    
    config["ckptEpochs"] = int(config["ckptEpochs"])

    if "device" not in config:
        raise Exception("Device not specified in config (>= 0 -> cuda ; < 0 -> cpu)")
    
    device = int(config["device"])

    if "addNoise" not in config:
        raise Exception("Whether noise should be added to the FFTs must be specified")
    else:
        if "noiseMean" not in config or "noiseDev" not in config:
            raise Exception("If noise addition is desired, the noise mean and stddev must be specified")
    
        config["noiseMean"] = float(config["noiseMean"])
        config["noiseDev"] = float(config["noiseDev"])

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
    
    print("Instantiating data loaders...\n")

    ds_train = NoteDataset(config, "train")
    
    no_train_samples = len(ds_train)

    train_data_loader = DataLoader(
        dataset=ds_train,
        batch_size=batch_size,
        num_workers=0
    )
    
    print(f"Total train dataset length: {no_train_samples} samples\n")

    ds_test = None

    if test_subdirs:
        ds_test = NoteDataset(config, "test")

    no_test_samples = len(ds_test)

    test_data_loader = DataLoader(
        dataset=ds_test,
        batch_size=2,
        num_workers=0
    )

    print(f"Total test dataset length: {no_test_samples} samples\n")

    print("Instantiating model...")

    model = NoteClassifier()
    optimizer = Adam(model.parameters(), learning_rate, betas=(beta1, beta2), eps=eps)
    criterion = CrossEntropyLoss()

    model.to(device).train()

    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        no_train_samples=no_train_samples,
        no_test_samples=no_test_samples,
        config=config
    )