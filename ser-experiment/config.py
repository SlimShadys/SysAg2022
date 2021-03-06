import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Configuration train phase")

parser.add_argument("-a", "--attention", type=str, default="bam", choices=["no", "se", "bam", "cbam"], help='Chose the attention module')
parser.add_argument("-bs", "--batch_size", type=int, default=64, help='Batch size to use for training')
parser.add_argument("-cp", "--checkpoint", type=str, default=None, help='Checkpoint model directory')
parser.add_argument("-cw", "--class_weights", type=str2bool, const=True, nargs='?', default=False, help='Use the class weights in loss function')
parser.add_argument("-d", "--dataset", type=str, default="demos", choices=["demos", "emovo", "demosemovo", "demosemovogender", "opera7"], help='Choose the dataset')
parser.add_argument("-e", "--epochs", type=int, default=100, help='Number of epochs')
parser.add_argument("-g", "--gender", type=str, default="male", choices=["male", "female", "all"], help='Choose the gender of the dataset')
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help='Learning rate to use for training')
parser.add_argument("-lm", "--loadModel", type=str, default=None, help='Best model directory')
parser.add_argument("-m", "--monitor", type=str, default="acc", choices=["acc", "loss"], help='Chose to monitor the validation accuracy or loss')
parser.add_argument("-nm", "--momentum", type=float, default=0, help='Value of momentum')
parser.add_argument("-o", "--optimizer", type=str, default="adam", choices=["adam", "sgd"], help='Chose the optimizer')
parser.add_argument("-p", "--patience", type=int, default=10, help='Number of epochs without improvements before reducing the learning rate')
parser.add_argument("-s", "--stats", type=str, default="no", choices=["no", "imagenet"], help='Chose the mean and standard deviation')
parser.add_argument("-se", "--seed", type=int, default=1, help='Random seed')
parser.add_argument("-ud","--uses_drive", type=str2bool, const=True, nargs='?', default=False, help='Whether using Drive for saving models / results')
parser.add_argument("-v", "--validation", type=str, default=None, choices=["gender", "emotion"], help='Choose on what to do validation')
parser.add_argument("-w", "--workers", type=int, default=0, help='Set number of workers')
parser.add_argument("-wa", "--withAugmentation", type=str2bool, const=True, nargs='?', default=False, help='Use augmentation files during training')
parser.add_argument("-wd", "--weight_decay", type=float, default=0, help='Value of weight decay')

args = parser.parse_args()
return_args = parser.parse_args()
