import logging
import os
import platform

import nni
import torch

if platform.system() == "Linux":
    import shutil

import warnings

from nni.utils import merge_parameter
from torchvision import transforms
from tqdm import tqdm

from config import args, return_args
from dataloader.opera7 import Opera7
from dataloader.emovo import Emovo
from dataloader.demos import Demos
from dataloader.demosemovo import DemosEmovo
from dataloader.demosemovogender import DemosEmovoGender
from models.bam.vggface2_bam import VGGFace2BAM
from models.cbam.vggface2_cbam import VGGFace2CBAM
from models.resnet50.vggface2 import VGGFace2
from models.se.vggface2_se import VGGFace2SE
from utility.checkpoint import load_model
from utility.utility import setup_seed 
from utility.confusion_matrix import show_confusion_matrix, get_classification_report

warnings.filterwarnings('ignore')

setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')

def main(args):

    # Args for debugging through IDE
    #args['dataset'] = 'demos'                                                          # Replace with you own dataset
    #args['gender'] = 'all'                                                             # Gender for the training dataset
    #args['validation'] = 'emotion'                                                     # Choose on what to do validation ("emotion" / "gender")
    #args['uses_drive'] = False                                                         # Whether to choose Drive to save results
    #args['loadModel'] = 'result/SysAg2022/best_model_gruppo_N_all.pt'                  # Load model for validation
    #args['attention'] = 'bam'                                                          # Choose your model type (Bam) / (CBam)
    #args['batch_size'] = 64                                                            # Batch size for training

    print("Starting validation with the following configuration:")
    print("Attention module: {}".format(args['attention']))
    print("Batch size: {}".format(args['batch_size']))
    print("Dataset: {}".format(args['dataset']))
    print("Gender: {}".format(args['gender']))
    print("Validation: {}".format(args['validation']))
    print("Load model: {}".format(args['loadModel']))
    print("Checkpoint model: {}".format(args['checkpoint']))
    print("Stats: {}".format(args['stats']))
    print("Uses Drive: {}".format(args["uses_drive"]))
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

        if not(os.path.exists("../../gdrive/MyDrive/SysAg2022/{}/{}/{}".format(args["dataset"], args["attention"], args["gender"]))):
           os.makedirs("../../gdrive/MyDrive/SysAg2022/{}/{}/{}".format(args["dataset"], args["attention"], args["gender"]))

        if not(os.path.exists("../../gdrive/MyDrive/SysAg2022/{}/{}/{}".format(args["dataset"], args["attention"], args["gender"]))):
           os.makedirs("../../gdrive/MyDrive/SysAg2022/{}/{}/{}".format(args["dataset"], args["attention"], args["gender"]))

    if not(os.path.exists(os.path.join("result", args["dataset"], args["attention"], args["gender"]))):
           os.makedirs(os.path.join("result", args["dataset"], args["attention"], args["gender"]))

    if(torch.cuda.is_available()):
        device = torch.device("cuda")
        print("===================================================")
        print('Cuda available: {}'.format(torch.cuda.is_available()))
        print("GPU: " + torch.cuda.get_device_name(torch.cuda.current_device()))
        print("Total memory: {:.1f} GB".format((float(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)))))
        print("===================================================")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print('Cuda not available. Using CPU.')

    if args["stats"] == "imagenet":
        # imagenet
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]

        val_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_mean, std=data_std),
        ])
    else:
        val_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    if args["dataset"] == "opera7":
        val_data = Opera7(gender=args["gender"], validation=args["validation"], split="val", transform=val_preprocess)
        if (args["validation"] == "gender"):
            classes = 2
            label_mapping = {
                0: "Donna",
                1: "Uomo",
            }
        else:
            classes = 7
            label_mapping = {
                0: "Disgusto",
                1: "Gioia",
                2: "Paura",
                3: "Rabbia",
                4: "Sorpresa",
                5: "Tristezza",
                6: "Neutrale"
            }
    elif args["dataset"]  == "emovo":
        val_data = Emovo(gender=args["gender"], validation=args["validation"], split="val", transform=val_preprocess)
        if (args["validation"] == "gender"):
            classes = 2
            label_mapping = {
                0: "Donna",
                1: "Uomo",
            }
        else:
            classes = 7
            label_mapping = {
                0: "Disgusto",
                1: "Gioia",
                2: "Paura",
                3: "Rabbia",
                4: "Sorpresa",
                5: "Tristezza",
                6: "Neutrale"
            }
    elif args["dataset"]  == "demos":
        val_data = Demos(gender=args["gender"], validation=args["validation"], split="val", transform=val_preprocess)
        if (args["validation"] == "gender"):
            classes = 2
            label_mapping = {
                0: "Donna",
                1: "Uomo",
            }
        else:
            classes = 7
            label_mapping = {
                0: "Disgusto",
                1: "Gioia",
                2: "Paura",
                3: "Rabbia",
                4: "Sorpresa",
                5: "Tristezza",
                6: "Neutrale"
            }
    elif args["dataset"]  == "demosemovo":
        val_data = DemosEmovo(gender=args["gender"], validation=args["validation"], split="val", transform=val_preprocess)
        if (args["validation"] == "gender"):
            classes = 2
            label_mapping = {
                0: "Donna",
                1: "Uomo",
            }
        else:
            classes = 7
            label_mapping = {
                0: "Disgusto",
                1: "Gioia",
                2: "Paura",
                3: "Rabbia",
                4: "Sorpresa",
                5: "Tristezza",
                6: "Neutrale"
            }
    elif args["dataset"]  == "demosemovogender":
        val_data = DemosEmovoGender(gender=args["gender"], validation=args["validation"], split="val", transform=val_preprocess)
        if (args["validation"] == "gender"):
            classes = 2
            label_mapping = {
                0: "Donna",
                1: "Uomo",
            }
        else:
            classes = 7
            label_mapping = {
                0: "Disgusto",
                1: "Gioia",
                2: "Paura",
                3: "Rabbia",
                4: "Sorpresa",
                5: "Tristezza",
                6: "Neutrale"
            }
    else:
        val_data = Opera7(gender=args["gender"], split="val", validation=args["validation"], transform=val_preprocess)
        if (args["validation"] == "gender"):
            classes = 2
            label_mapping = {
                0: "Donna",
                1: "Uomo",
            }
        else:
            classes = 7
            label_mapping = {
                0: "Disgusto",
                1: "Gioia",
                2: "Paura",
                3: "Rabbia",
                4: "Sorpresa",
                5: "Tristezza",
                6: "Neutrale"
            }
    
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

    if args['loadModel']:
        print("You specified a pre-loading directory for a model.")
        print("The directory is: {}".format(args["loadModel"]))
        if os.path.isfile(args["loadModel"]):
            print("=> Loading model '{}'".format(args["loadModel"]))
            model = load_model(args["loadModel"], model, device)
            print("Custom model loaded successfully")
        else:
            print("=> No model found at '{}'".format(args["loadModel"]))
            print("Are you sure the directory / model exist?")
            exit(0)
        print("-------------------------------------------------------")
    else:
        print("Default {} model loaded.".format(args["attention"]))

    # validate the model
    model.eval()
    
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args["batch_size"], shuffle=True, num_workers=args["workers"])

    y_true = []
    y_pred = []
    labels_list = []

    val_correct = 0

    print("\nStarting validation...")
    for batch_idx, batch in enumerate(tqdm(val_loader)):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            val_outputs = model(images)
            _, val_preds = torch.max(val_outputs, 1)
            val_correct += torch.sum(val_preds == labels.data)
                
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(val_preds.detach().cpu().numpy().tolist())

    for i in range(len(label_mapping)):
        labels_list.append(label_mapping[i])

    print("Num correct: {}".format(val_correct))
    print("Num samples: {}".format(len(val_data)))

    val_acc = (val_correct.double() / len(val_data)) * 100

    delimiter = "\n===================================================================================\n"
    write = F'Accuracy of the network on the test images: {val_acc:.3f}%'
    print(F'\n{write}')

    classificationReport = get_classification_report(y_true, y_pred, labels_list)
    print(classificationReport)

    # Scriviamo i risultati in un file testuale
    f = open("result/{}/{}/{}/res_validation_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]), "w")
    
    f.write(write)
    f.write(delimiter)
    f.write("Modello utilizzato: {}".format(args['loadModel']))
    f.write(delimiter)
    f.write("val_correct: {val_correct} || val_samples: {val_samples}".format(
        val_correct = val_correct,
        val_samples = len(val_data)))
    f.write(delimiter)           
    f.write(classificationReport)
    f.close()

    show_confusion_matrix(y_true, y_pred, labels_list, "result/{}/{}/{}/".format(args["dataset"], args["attention"], args["gender"]))

    if platform.system() == "Linux" and args['uses_drive']:
        shutil.copy("result/{}/{}/{}/res_validation_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]), "../../gdrive/MyDrive/SysAg2022/{}/{}/{}/res_validation_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]))
        shutil.copy("result/{}/{}/{}/confusion_matrix.png".format(args["dataset"], args["attention"], args["gender"]), "../../gdrive/MyDrive/SysAg2022/{}/{}/{}/confusion_matrix.png".format(args["dataset"], args["attention"], args["gender"]))

    print("===================================Testing Finished===================================")

if __name__ == '__main__':
      
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))

    main(params)
