import torch 
import numpy as np
import random
import os
import shutil


def to_categorical(value, limit=300):
    value = value.float()  # Avoid any fp16 shenanigans
    value = value.clamp(-limit, limit)
    distribution = torch.zeros(value.shape[0], (limit * 2 + 1),
                               device=value.device)
    lower = value.floor().long() + limit
    upper = value.ceil().long() + limit
    upper_weight = value % 1
    lower_weight = 1 - upper_weight
    distribution.scatter_add_(-1, lower.unsqueeze(-1),
                              lower_weight.unsqueeze(-1))
    distribution.scatter_add_(-1, upper.unsqueeze(-1),
                              upper_weight.unsqueeze(-1))
    return distribution


def maybe_transform(image, transform, alt_transform, p=0.8):
    processed_images = transform(image)
    if p >= 1:
        return processed_images
    else:
        base_images = alt_transform(image)
        mask = torch.rand((processed_images.shape[0], 1, 1, 1),
                          device=processed_images.device)
        mask = (mask < p).float()
        processed_images = mask * processed_images + (1 - mask) * base_images
        return processed_images


def infer_leading_dims(tensor, dim):
    """Looks for up to two leading dimensions in ``tensor``, before
    the data dimensions, of which there are assumed to be ``dim`` number.
    For use at beginning of model's ``forward()`` method, which should 
    finish with ``restore_leading_dims()`` (see that function for help.)
    Returns:
    lead_dim: int --number of leading dims found.
    T: int --size of first leading dim, if two leading dims, o/w 1.
    B: int --size of first leading dim if one, second leading dim if two, o/w 1.
    shape: tensor shape after leading dims.
    """
    lead_dim = tensor.dim() - dim
    assert lead_dim in (0, 1, 2)
    if lead_dim == 2:
        T, B = tensor.shape[:2]
    else:
        T = 1
        B = 1 if lead_dim == 0 else tensor.shape[0]
    shape = tensor.shape[lead_dim:]
    return lead_dim, T, B, shape


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path, load):
    if not load:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def update_state_dict(model, state_dict, tau=1, strip_ddp=True):
    """Update the state dict of ``model`` using the input ``state_dict``, which
    must match format.  ``tau==1`` applies hard update, copying the values, ``0<tau<1``
    applies soft update: ``tau * new + (1 - tau) * old``.
    """
    if strip_ddp:
        state_dict = strip_ddp_state_dict(state_dict)
    if tau == 1:
        model.load_state_dict(state_dict)
    elif tau > 0:
        update_sd = {
            k: tau * state_dict[k] + (1 - tau) * v
            for k, v in model.state_dict().items()
        }
        model.load_state_dict(update_sd)


def strip_ddp_state_dict(state_dict):
    """ Workaround the fact that DistributedDataParallel prepends 'module.' to
    every key, but the sampler models will not be wrapped in
    DistributedDataParallel. (Solution from PyTorch forums.)"""
    clean_state_dict = type(state_dict)()
    for k, v in state_dict.items():
        key = k[7:] if k[:7] == "module." else k
        clean_state_dict[key] = v
    return clean_state_dict


def search_last_step(search_dir):
    """
    search in a directory, find last step from file names
    """
    file_list = [f for f in os.listdir(search_dir) if os.path.isfile(os.path.join(search_dir, f))]
    name_list = sorted([x.split('.')[0] for x in file_list])
    num_list = [int(x.split("_")[-1]) for x in name_list]
    return max(num_list)


if __name__ == "__main__":
    search_last_step('./')