import torch
import shutil


def save_checkpoint(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir + '/checkpoint.pt'
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + '/best_model.pt'
        shutil.copyfile(f_path, best_fpath)


def load_checkpoint(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['epoch']


def load_model(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model
