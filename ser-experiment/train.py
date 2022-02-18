import logging
import os
import platform

import nni
import torch

if platform.system() == "Linux":
    import shutil

import warnings

import torch.nn as nn
import torch.optim as optim
from nni.utils import merge_parameter
from torchvision import transforms
from tqdm import tqdm

from config import args, return_args
from dataloader.demos import Demos
from dataloader.demosemovo import DemosEmovo
from dataloader.emovo import Emovo
from models.bam.vggface2_bam import VGGFace2BAM
from models.cbam.vggface2_cbam import VGGFace2CBAM
from models.resnet50.vggface2 import VGGFace2
from models.se.vggface2_se import VGGFace2SE
from utility.checkpoint import load_checkpoint, save_checkpoint
from utility.utility import setup_seed

warnings.filterwarnings('ignore')

setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')

def main(args):

    print("Starting training with the following configuration:")
    print("Attention module: {}".format(args['attention']))
    print("Batch size: {}".format(args['batch_size']))
    print("Class weights: {}".format(args['class_weights']))
    print("Dataset: {}".format(args['dataset']))
    print("Epochs: {}".format(args['epochs']))
    print("Gender: {}".format(args['gender']))
    print("Learning rate: {}".format(args['learning_rate']))
    print("Metric to monitor: {}".format(args['monitor']))
    print("Momentum: {}".format(args['momentum']))
    print("Optimizer: {}".format(args['optimizer']))
    print("Patience: {}".format(args['patience']))
    print("Checkpoint model: {}".format(args['checkpoint']))
    print("Stats: {}".format(args['stats']))
    print("Uses Drive: {}".format(args["uses_drive"]))
    print("Weight decay: {}".format(args['weight_decay']))
    print("With Augmentation: {}".format(args["withAugmentation"]))
    print("Workers: {}".format(args['workers']))
    
    if platform.system() == "Linux" and args['uses_drive']:
        print("----------------------------")
        print("** Google Drive Sign In **")
        if not(os.path.exists("../../gdrive/")):
            print("No Google Drive path detected! Please mount it before running this script or disable ""uses_drive"" flag!")
            print("----------------------------")
            exit(0)
        else:
            print("** Successfully logged in! **")
            print("----------------------------")

        if not(os.path.exists("../../gdrive/MyDrive/SysAg2022/{}/{}/{}/{}".format(args["dataset"], args["attention"], args["gender"], "checkpoint"))):
           os.makedirs("../../gdrive/MyDrive/SysAg2022/{}/{}/{}/{}".format(args["dataset"], args["attention"], args["gender"], "checkpoint"))

        if not(os.path.exists("../../gdrive/MyDrive/SysAg2022/{}/{}/{}/{}".format(args["dataset"], args["attention"], args["gender"], "best"))):
           os.makedirs("../../gdrive/MyDrive/SysAg2022/{}/{}/{}/{}".format(args["dataset"], args["attention"], args["gender"], "best"))

    if not(os.path.exists(os.path.join("result", args["dataset"], args["attention"], args["gender"], "checkpoint"))):
           os.makedirs(os.path.join("result", args["dataset"], args["attention"], args["gender"], "checkpoint"))

    if not(os.path.exists(os.path.join("result", args["dataset"], args["attention"], args["gender"], "best"))):
           os.makedirs(os.path.join("result", args["dataset"], args["attention"], args["gender"], "best"))

    if(torch.cuda.is_available()):
        device = torch.device("cuda")
        print('Cuda available: {}'.format(torch.cuda.is_available()))
        print("GPU: " + torch.cuda.get_device_name(torch.cuda.current_device()))
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print('Cuda not available. Using CPU.')

    if args["stats"] == "imagenet":
        # imagenet
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]
    
        train_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_mean, std=data_std),
        ])
    
        val_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_mean, std=data_std),
        ])
    else: # no
        train_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    
        val_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    if args["dataset"] == "demos":
        train_data = Demos(gender=args["gender"], split="train", transform=train_preprocess, withAugmentation=args["withAugmentation"])
        val_data = Demos(gender=args["gender"], split="val", transform=val_preprocess, withAugmentation=args["withAugmentation"])
        # TODO: verificare che il numero di classi sia corretto
        classes = 7
    elif args["dataset"]  == "emovo":
        train_data = Emovo(gender=args["gender"], split="train", transform=train_preprocess, withAugmentation=args["withAugmentation"])
        val_data = Emovo(gender=args["gender"], split="val", transform=val_preprocess, withAugmentation=args["withAugmentation"])
        # TODO: verificare che il numero di classi sia corretto
        classes = 7
    elif args["dataset"]  == "demosemovo":
        train_data = DemosEmovo(gender=args["gender"], split="train", transform=train_preprocess, withAugmentation=args["withAugmentation"])
        val_data = DemosEmovo(gender=args["gender"], split="val", transform=val_preprocess, withAugmentation=args["withAugmentation"])
        # TODO: verificare che il numero di classi sia corretto
        classes = 7
    else:
        train_data = Demos(gender=args["gender"], split="train", transform=train_preprocess, withAugmentation=args["withAugmentation"])
        val_data = Demos(gender=args["gender"], split="val", transform=val_preprocess, withAugmentation=args["withAugmentation"])
        # TODO: verificare che il numero di classi sia corretto
        classes = 7
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args["batch_size"], shuffle=True, num_workers=args["workers"])
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args["batch_size"], shuffle=True, num_workers=args["workers"])
    
    if args["attention"] == "no":
        model = VGGFace2(pretrained=False, classes=classes).to(device)
    elif args["attention"] == "se":
        model = VGGFace2SE(classes=classes).to(device)
    elif args["attention"] == "bam":
        model = VGGFace2BAM(classes=classes).to(device)
    elif args["attention"] == "cbam":
        model = VGGFace2CBAM(classes=classes).to(device)
    else:
        model = VGGFace2(pretrained=False, classes=classes).to(device)
    
    print("Model archticture: ", model)
    
    start_epoch = 0
    
    best_val_loss = 1000000
    best_val_acc = 0
    
    criterion = nn.CrossEntropyLoss()
    
    if args["optimizer"] == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args["learning_rate"],
                              momentum=args["momentum"], weight_decay=args["weight_decay"])
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args["learning_rate"],
                               weight_decay=args["weight_decay"])
    
    if args["monitor"] == "loss":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args["patience"], verbose=True)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args["patience"], mode="max", verbose=True)
    
    if args['checkpoint']:
        print("You specified a pre-loading directory for checkpoints.")
        print("The directory is: {}".format(args["checkpoint"]))
        if os.path.isfile(args["checkpoint"]):
            print("=> Loading checkpoint '{}'".format(args["checkpoint"]))
            model, optimizer, scheduler, epoch, val_loss, val_acc = load_checkpoint(args["checkpoint"], model, optimizer, scheduler)
            start_epoch = epoch
            best_val_loss = val_loss
            best_val_acc = val_acc
            print("- Epoch loaded : {}".format(start_epoch))
            print("- Best validation loss loaded     : {:.8f}".format(best_val_loss))
            print("- Best validation accuracy loaded : {:.8f}".format(best_val_acc))
            print("Checkpoint loaded successfully")
        else:
            print("=> No checkpoint found at '{}'".format(args["checkpoint"]))
            print("Are you sure the directory / checkpoint exist?")
        print("-------------------------------------------------------")

    for e in range(start_epoch, args["epochs"]):
        train_loss = 0
        validation_loss = 0
        train_correct = 0
        val_correct = 0
        is_best = False
    
        print(F'Starting Epoch n.{e+1}')
        print(F"\t- {e * len(train_loader.dataset)} samples processed until now at a learning rate of {args['learning_rate']}")
        print("-------------------------------------------------------")

        # train the model
        model.train()
        
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
    
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
    
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct.double() / len(train_data)
    
        # validate the model
        model.eval()
            
        print("\nStarting validation...")
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
    
            with torch.no_grad():
                val_outputs = model(images)
                val_loss = criterion(val_outputs, labels)
                validation_loss += val_loss.item()
                _, val_preds = torch.max(val_outputs, 1)
                val_correct += torch.sum(val_preds == labels.data)
                    
        validation_loss = validation_loss / len(val_loader)
        val_acc = val_correct.double() / len(val_data)
    
        if args["monitor"] == "loss":
            scheduler.step(validation_loss)
    
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                is_best = True
        else:
            scheduler.step(val_acc)
    
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                is_best = True
    
        checkpoint = {
            'epoch': e + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc
        }
        
        save_checkpoint(checkpoint, is_best, "result/{}/{}/{}/checkpoint".format(args["dataset"], args["attention"], args["gender"]), "result/{}/{}/{}/best".format(args["dataset"], args["attention"], args["gender"]), args["gender"], e+1)
    
        if is_best:
            write = 'Epoch[{0}/{1}]\tTraining loss: {training_loss:.8f}\tValidation Loss {validation_loss:.8f}\tTraining Accuracy {training_accuracy:.3f}% \tValidation Accuracy {validation_accuracy:.3f}% [saved]\n'.format(e + 1, args['epochs'], training_loss=train_loss, validation_loss=validation_loss, training_accuracy=train_acc * 100, validation_accuracy=val_acc * 100)
            print("\n{}".format(write))

            if platform.system() == "Linux" and args['uses_drive']:
                shutil.copy("result/{}/{}/{}/best".format(args["dataset"], args["attention"], args["gender"]) + "/best_model_{}-epoch_{}.pt".format(args["gender"], e+1), "../../gdrive/MyDrive/SysAg2022/{}/{}/{}/{}/best_model_{}-epoch_{}.pt".format(args["dataset"], args["attention"], args["gender"], "best", args["gender"], e+1))
        else:
            write = 'Epoch[{0}/{1}]\tTraining loss: {training_loss:.8f}\tValidation Loss {validation_loss:.8f}\tTraining Accuracy {training_accuracy:.3f}% \tValidation Accuracy {validation_accuracy:.3f}%\n'.format(e + 1, args['epochs'], training_loss=train_loss, validation_loss=validation_loss, training_accuracy=train_acc * 100, validation_accuracy=val_acc * 100)
            print("\n{}".format(write))
            
            if platform.system() == "Linux" and args['uses_drive']:
                shutil.copy("result/{}/{}/{}/checkpoint".format(args["dataset"], args["attention"], args["gender"]) + "/checkpoint_{}-epoch_{}.pt".format(args["gender"], e+1), "../../gdrive/MyDrive/SysAg2022/{}/{}/{}/{}/checkpoint_{}-epoch_{}.pt".format(args["dataset"], args["attention"], args["gender"], "checkpoint", args["gender"], e+1))
            
        # Scriviamo i risultati in un file testuale
        if platform.system() == "Linux" and args['uses_drive']:

            if(not os.path.exists("../../gdrive/MyDrive/SysAg2022/{}/{}/{}/res_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]))):

                if(not os.path.exists("result/{}/{}/{}/res_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]))):
                    f = open("result/{}/{}/{}/res_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]), "w")
                else:
                    f = open("result/{}/{}/{}/res_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]), "a")
                
                f.write(write)
                f.close()

                shutil.copy("result/{}/{}/{}/res_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]), "../../gdrive/MyDrive/SysAg2022/{}/{}/{}/res_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]))
            else:
                f = open("../../gdrive/MyDrive/SysAg2022/{}/{}/{}/res_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]), "r")
                lines = f.readlines()
                f.close()

                f2 = open("result/{}/{}/{}/res_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]), "w")  

                # Riscriviamo tutte le righe precedenti
                # Sarebbe molto più semplice aprire il file .txt da Google Drive stesso,
                # ma ciò comporterebbe un errore di scrittura IO (UnsupportedOperation - Python)
                for line in lines:
                    f2.write(line)
                
                # Scriviamo la nuova riga relativa all'epoch n
                f2.write(write)
                f2.close()

                shutil.copy("result/{}/{}/{}/res_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]), "../../gdrive/MyDrive/SysAg2022/{}/{}/{}/res_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]))
        else:
            if(not os.path.exists("result/{}/{}/{}/res_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]))):
                f = open("result/{}/{}/{}/res_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]), "w")
            else:
                f = open("result/{}/{}/{}/res_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]), "a")
            f.write(write)
            f.close()

        print("------------------------------------------------------")
    
    print("===================================Training Finished===================================")

if __name__ == '__main__':
      
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))

    main(params)
