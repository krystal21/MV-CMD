import torch


def save_model(model, optimizer, epoch, save_file):
    print("==> Saving...")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, save_file)
    del state
