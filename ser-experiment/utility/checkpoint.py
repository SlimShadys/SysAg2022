import torch
import shutil

def save_checkpoint(state, is_best, checkpoint_dir, best_model_dir, gender, epoch):
    f_path = checkpoint_dir + '/checkpoint_{}-epoch_{}.pt'.format(gender,epoch)
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + '/best_model_{}-epoch_{}.pt'.format(gender,epoch)
        shutil.copyfile(f_path, best_fpath)


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
    return model
