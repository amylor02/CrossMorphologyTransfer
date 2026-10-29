"""Optional foot-contact conditioning for the autoregressive prior.

Each actual foot has one binary contact channel (1 contact, 0 no contact) and one validity channel.
Unavailable or masked feet have contact 0 and validity 0, so a missing channel is never read as swing.
"""

import torch


def foot_count(feet_idx):
    # duplicated indices (for example [3, 7, 3, 7] in biped configurations) count as one foot
    return len(dict.fromkeys(feet_idx))


def sample_foot_keep_mask(batch_size, n_feet, mask_prob, device):
    # true keeps a foot for the whole sequence; false masks that foot channel in every frame
    return torch.rand(batch_size, n_feet, device=device) >= mask_prob


def apply_foot_keep_mask(contact, valid, keep):
    # contact, valid: [B, T, N]; keep: [B, N]
    keep = keep.unsqueeze(1)
    zero = torch.zeros_like(contact)
    contact = torch.where(keep, contact, zero)
    valid = torch.where(keep, valid, zero)
    return contact, valid


def foot_contact_channels(contact, valid):
    # contact and validity are replaced, not multiplied, so raw values under invalid feet never reach the prior
    zero = torch.zeros_like(contact)
    contact = torch.where(valid > 0.5, contact, zero)
    return torch.cat([contact, valid], dim=-1)


def map_source_foot_contact(source_contact, target_frames, n_target):
    # the first min(Ns, Nt) source feet drive the target in configured order.
    # extra target feet and frames beyond the source are invalid. extra source feet are discarded.
    frames = min(target_frames, source_contact.shape[0])
    n = min(source_contact.shape[1], n_target)
    contact = source_contact.new_zeros((target_frames, n_target))
    valid = source_contact.new_zeros((target_frames, n_target))
    contact[:frames, :n] = source_contact[:frames, :n]
    valid[:frames, :n] = 1.0
    return contact, valid
