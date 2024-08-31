import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, jit

"""
NOTE: here we provide both 
1) the for-loop based implementation to show the logic of the proposed temporal contrastive loss to match the equation, and
2) the efficient implementation of it
"""

def linspace_with_skip(start, skip_size, num_elements, device):
    end = start + skip_size * (num_elements - 1)
    return torch.linspace(start, end, num_elements, device=device)


@jit.script
def logsumexp(x: Tensor, dim: int) -> Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))


def temp_contrastive(similarities, T, min_temp, max_levels=6, skip=0.075):
    
    def num_levels(x, T):
        return math.ceil(T / 2) + abs(T // 2 - x) + (T % 2 == 0 and x >= T // 2)

    BT = similarities.size(0)
    B = BT // T
    loss = 0
    # base distance tensor
    base_distances = torch.arange(T, device=similarities.device)
    batch_base_distances = torch.arange(B, device=similarities.device).repeat_interleave(T)

    # for each query
    for query_idx, bt_scores in enumerate(similarities):
        # focus on the logits for [[],[],...,[...,bt_score,...],[],...,[]]
        curr_b, curr_t = query_idx // T, query_idx % T  # check relative batch number and temporal position
        # count how many of levels are permitted
        curr_num_levels = num_levels(curr_t, T)
        levels = min(curr_num_levels, max_levels)
        end = min_temp + skip * (max_levels - 1)
        temp_selector = torch.linspace(min_temp, end, max_levels, device=similarities.device)
        # calculate each level and construct ranking list  
        for level_idx in range(levels):
            # Calculate relative distances from the reference index t
            relative_distances = torch.abs(base_distances - curr_t)
            # Get mask for positives samples
            positive_mask = relative_distances == level_idx
            # Get mask for negative samples within current temporal block
            negative_mask = relative_distances >= level_idx
            # Get mask for negative samples in all other temporal batches
            negative_mask_others = batch_base_distances != curr_b
            # Get positives
            positives = bt_scores[curr_b*T:(curr_b+1)*T][positive_mask]
            # Get negatives 
            negatives = bt_scores[curr_b*T:(curr_b+1)*T][negative_mask]
            other_negatives = bt_scores[negative_mask_others]
            all_negatives = torch.cat([negatives, other_negatives])
            # compute current level loss
            # current level tau (temperature)
            curr_level_temp = temp_selector[level_idx]
            l_pos = positives / curr_level_temp
            l_neg = all_negatives / curr_level_temp
            # Use logsumexp for numerical stability
            logsumexp_l_pos = torch.logsumexp(l_pos, dim=0)
            logsumexp_l_neg = torch.logsumexp(l_neg, dim=0)
            # Compute log(curr_score) as the difference of logsumexp values
            log_curr_score = logsumexp_l_pos - logsumexp_l_neg
            # current level ranked contrastive Loss computation
            loss -= log_curr_score 

    loss = loss / BT / max_levels   # average loss across all elements

    return loss


def temp_contrastive_efficient(similarities, T, min_temp, max_levels=6, skip=0.075):
    assert max_levels <= T, 'max_level should be smaller or equal to temporal size'
    B, BT = similarities.size(0) // T, similarities.size(0)

    # Precompute levels and temp_selectors outside of the loop
    arange_T = torch.arange(T, device=similarities.device)
    arange_B = torch.arange(B, device=similarities.device)
    levels = (T + 1) // 2 + torch.abs(T // 2 - arange_T) + (T % 2 == 0) * (arange_T >= T // 2)
    
    # Generate linearly spaced temperatures for all possible levels
    all_temps = linspace_with_skip(min_temp, skip, max_levels, device=similarities.device)

    # Calculate the indices for each level in the range [0, max_levels)
    level_indices = torch.arange(max_levels, device=similarities.device).unsqueeze(0).expand(T, -1)

    # Create a mask where the level_indices are less than the adjusted levels
    mask = level_indices < levels.unsqueeze(-1)

    # Initialize temp_selectors tensor, and apply the mask to assign temperatures
    temp_selectors = torch.where(mask, all_temps, -1)

    # Create batch and time index tensors
    batch_indices = arange_B.view(-1, 1).expand(-1, T).reshape(-1)
    time_indices = arange_T.repeat(B)

    # Calculate relative distances once: [BT, BT, 1]
    relative_distances = (time_indices.unsqueeze(-1) - time_indices.unsqueeze(0)).abs().unsqueeze(-1)

    # Select tempeature from levels
    temp = temp_selectors.T.repeat(1, B)
    temp_mask = (temp == -1).T
    expanded_temp_mask = temp_mask.unsqueeze(1).expand(-1, BT, -1)    # [BT, BT, L]
    
    # Apply temperature scaling: [BT, 1]
    scaled_similarities = torch.div(similarities.unsqueeze(-1), temp.T.unsqueeze(1))
    scaled_similarities[expanded_temp_mask] = float('nan')

    # Create masks for all queries at once: [BT, BT]
    positive_mask = (relative_distances == arange_T)[...,:max_levels] & (batch_indices == arange_B.unsqueeze(-1)).unsqueeze(1).expand(-1, T, -1).reshape(BT, BT, 1)
    negative_mask = (relative_distances >= arange_T)[...,:max_levels] | (batch_indices != arange_B.unsqueeze(-1)).unsqueeze(1).expand(-1, T, -1).reshape(BT, BT, 1)
    positive_mask = positive_mask & (~scaled_similarities.isnan())
    negative_mask = negative_mask & (~scaled_similarities.isnan())

    # Compute logsumexp for positives and negatives: should be [BT, L]
    logsumexp_pos = torch.logsumexp(torch.where(positive_mask, scaled_similarities, float('-inf')), dim=1)
    logsumexp_neg = torch.logsumexp(torch.where(negative_mask, scaled_similarities, float('-inf')), dim=1)

    # Accumulate the loss
    loss = torch.nanmean(logsumexp_neg - logsumexp_pos)

    return loss