from biomodel_handler.data_loader import slice_single_fasta_sequence, load_net
from biomodel_handler.net_classes import Multiple_Input_Model
from biomodel_handler.bio_plot import plot_weights
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
import numpy as np
import torch


########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].


# ----- Settings part -----


DEVICE_ID = 0

LABEL_POS = 1
LABEL_NEG = 0
SEQ_LENGTH = 1000


# ----- Working with DATA functions part -----


def decode_one_hot_class_labels(tensor_with_true_scores):
    vector_with_labels = []
    pos_idx = 1
    neg_idx = 0
    for row_idx in range(0, tensor_with_true_scores.size()[0]):
        if tensor_with_true_scores[row_idx, pos_idx].item() == 1.0:
            vector_with_labels.append(1.0)
        else:
            vector_with_labels.append(0.0)
    vector_with_labels = np.array(vector_with_labels)
    vector_with_labels = vector_with_labels.astype(int)
    return vector_with_labels


def get_scores_pos_class_labels(tensor_with_true_scores):
    vector_with_labels = []
    pos_idx = 1
    neg_idx = 0
    for row_idx in range(0, tensor_with_true_scores.size()[0]):
        vector_with_labels.append(tensor_with_true_scores[row_idx, pos_idx].item())

    return np.array(vector_with_labels)


# --- create np array with zeros --- #
def one_hot_encode_along_channel_axis(sequence):
    to_return = np.zeros((len(sequence), 4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return,
                                 sequence=sequence, one_hot_axis=1)
    return to_return


def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis == 0 or one_hot_axis == 1
    if one_hot_axis == 0:
        assert zeros_array.shape[1] == len(sequence)
    elif one_hot_axis == 1:
        assert zeros_array.shape[0] == len(sequence)
    # will mutate zeros_array
    for (i, char) in enumerate(sequence):
        if char == "A" or char == "a":
            char_idx = 0
        elif char == "C" or char == "c":
            char_idx = 1
        elif char == "G" or char == "g":
            char_idx = 2
        elif char == "T" or char == "t":
            char_idx = 3
        elif char == "N" or char == "n":
            continue  # leave that pos as all 0's
        else:
            raise RuntimeError("Unsupported character: " + str(char))
        if one_hot_axis == 0:
            zeros_array[char_idx, i] = 1
        elif one_hot_axis == 1:
            zeros_array[i, char_idx] = 1


def get_single_3d_feature_matrix(path_to_hg, chr, pos, nucleo):
    # кол-во нуклеотидов в одном сэмпле (1000 в нашем случае)
    num_nucleotides = SEQ_LENGTH

    seqs, ids = slice_single_fasta_sequence(chr, pos, nucleo, path_to_hg)
    if len(seqs[0]) != 1000:
        raise RuntimeError("Unsupported amount of features")

    # кол-во последовательностей в файле
    n_rows = len(ids)

    # создание трёхмерного массива:
    # x: кол-во последовательностей в файле
    # y: кол-во нуклеотидов в одном сэмпле
    # z: 4 - ACGT
    X_data_matrix = np.zeros((n_rows, num_nucleotides, 4))

    # функция enumerate генерирует кортежи, состоящие из двух элементов - индекса элемента и самого элемента
    # форматирование в бинарный формат файла
    for i, fasta_seq in enumerate(seqs):
        one_hot_encoded = one_hot_encode_along_channel_axis(fasta_seq)
        X_data_matrix[i, :, :] = one_hot_encoded

    # если есть NaN
    X_data_matrix[np.isnan(X_data_matrix)] = 0

    return X_data_matrix


def output_tensor_to_1d_vector(attributions):
    l = attributions.shape[2]
    res = []
    for seq in attributions:
        for i in range(l):
            if seq[0][i] != 0:
                res.append(seq[0][i])
            elif seq[1][i] != 0:
                res.append(seq[1][i])
            elif seq[2][i] != 0:
                res.append(seq[2][i])
            elif seq[3][i] != 0:
                res.append(seq[3][i])
    return np.array(res)


# ----- Main part -----


def get_z_scores(attributions):
    output_vect = output_tensor_to_1d_vector(attributions)
    z_scores = np.array(zscore(output_vect))
    return z_scores


def plot_distplot(z_scores, pic_name_unique):
    plot = sns.distplot(z_scores)
    plot.figure.savefig(r"user_media/dist_" + pic_name_unique + ".png")
    plt.clf()


def plot_barplot(z_scores, pic_name_unique):
    plt.bar(np.arange(1, 1001), z_scores)
    plt.savefig(r"user_media/bar_" + pic_name_unique + ".png")
    plt.clf()


def plot_bioinf_weights(attributions, pic_name_unique):
    fig = plot_weights(attributions, subticks_frequency=10, figsize=(50, 5))
    fig.savefig(r"user_media/bio_" + pic_name_unique + ".png")
    plt.close(fig)
    fig.clf()
    plt.clf()


def get_single_experiment_scores(path_to_hg, path_to_net, chr, pos, nucleo, pic_name_unique):

    # --- Preparing tensor from the input file ---

    X_input = get_single_3d_feature_matrix(path_to_hg, chr, pos, nucleo)
    X_input = torch.from_numpy(X_input)
    X_input = X_input.float()
    X_input = X_input.permute(0, 2, 1).contiguous()

    # --- Preparing the model ---

    use_gpu = torch.cuda.is_available()

    CNN = Multiple_Input_Model()

    if use_gpu:
        CNN = torch.nn.DataParallel(CNN, device_ids=[0])
        X_input = X_input.to(DEVICE_ID)

    # --- Loading net and getting metrics ---

    CNN, start_epoch, valid_loss_min = load_net(path_to_net, CNN)

    CNN.eval()

    ig = IntegratedGradients(CNN)
    # ig = DeepLift(CNN)
    attributions, delta = ig.attribute(X_input, target=1, return_convergence_delta=True)

    attributions = attributions.cpu().numpy()

    z_scores = get_z_scores(attributions)

    plot_barplot(z_scores, pic_name_unique)
    plot_distplot(z_scores, pic_name_unique)
    plot_bioinf_weights(attributions, pic_name_unique)

    most_features = []
    for i in range(len(z_scores)):
        if abs(z_scores[i] > 3):
            most_features.append(pos - 500 + i)

    return most_features
