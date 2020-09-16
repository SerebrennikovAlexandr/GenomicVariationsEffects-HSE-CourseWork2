import torch
import os


# ----- Main part -----


# Загрузка последовательности
def update_sequence(postfix, path_to_hg):
    fin = open(path_to_hg + postfix + ".fasta", 'r')
    seq = "".join(str(x)[:-1] for x in fin.readlines())
    seq = seq[len("assembly") + seq.find("assembly"):]
    fin.close()

    return seq


def slice_single_fasta_sequence(chr, pos, nucleo, path_to_hg):
    ids = []
    seqs = []

    ids.append(chr)

    chr_seq = update_sequence(chr, path_to_hg)

    mutation_pos = pos - 1
    mutation_seq = chr_seq[:mutation_pos] + nucleo + chr_seq[pos:]
    mutation_seq = mutation_seq[mutation_pos - 500:mutation_pos + 500]
    seqs.append(mutation_seq)

    return seqs, ids


def load_net(checkpoint_fpath, model):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, epoch value, min validation loss
    return model, checkpoint['epoch'], valid_loss_min
