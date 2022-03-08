import torch
import shutil

def save_checkpoint(state, is_best, checkpoint_dir, best_model_dir, gender, epoch):
    if is_best:
        path = best_model_dir + '/best_model_{}-epoch_{}.pt'.format(gender,epoch)
    else:
        path = checkpoint_dir + '/checkpoint_{}-epoch_{}.pt'.format(gender,epoch)
    torch.save(state, path)

def load_checkpoint(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    try:
        best_val_loss = checkpoint['best_val_loss']
    except:
        best_val_loss = 1000000
        pass

    try:
        best_val_acc = checkpoint['best_val_acc']
    except:
        best_val_acc = 0
        pass

    return model, optimizer, scheduler, checkpoint['epoch'], best_val_loss, best_val_acc


def load_model(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    # Val correct
    try:
        print("- Val correct loaded : {}".format(checkpoint['val_correct']))
    except:
        print("- No val correct present in this model")
        pass

    # Val samples
    try:
        print("- Val samples loaded : {}".format(checkpoint['val_samples']))
    except:
        print("- No val samples present in this model")
        pass

    return model
