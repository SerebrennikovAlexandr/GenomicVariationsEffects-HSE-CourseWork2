import torch
import numpy as np
import matplotlib
from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder
import sys
import torch.utils.data as data_utils
import torch.nn as nn
from captum.attr import IntegratedGradients, DeepLift
from conv_pytorch_binary_classifier import Multiple_Input_Model
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt


########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].


# ----- Settings part -----


DEVICE_ID = 0


LABEL_POS = 1
LABEL_NEG = 0
SEQ_LENGTH = 1000
LEARNING_RATE = 0.15
BATCH_SIZE = 32

PATH_TO_INPUT_FILE = r"gen_samples_data/H3K4me3/input.fasta"
PATH_TO_SAVE_BEST = r"../model_results/H3K4me3/best_model/best_checkpoint.pt"


# ----- Bio graph part -----


def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.2, 0.0],
        ]),
        np.array([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.8, 0.0],
        ]),
        np.array([
            [0.225, 0.45],
            [0.775, 0.45],
            [0.85, 0.3],
            [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.825, base + 0.085 * height], width=0.174, height=0.415 * height,
                                     facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.625, base + 0.35 * height], width=0.374, height=0.15 * height,
                                     facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 0.4, base],
                                              width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base + 0.8 * height],
                                              width=1.0, height=0.2 * height, facecolor=color, edgecolor=color,
                                              fill=True))


default_colors = {0: 'green', 1: 'blue', 2: 'orange', 3: 'red'}
default_plot_funcs = {0: plot_a, 1: plot_c, 2: plot_g, 3: plot_t}


def plot_weights_given_ax(ax, array,
                          height_padding_factor,
                          length_padding,
                          subticks_frequency,
                          highlight,
                          colors=default_colors,
                          plot_funcs=default_plot_funcs):
    if len(array.shape) == 3:
        array = np.squeeze(array)
    assert len(array.shape) == 2, array.shape
    if (array.shape[0] == 4 and array.shape[1] != 4):
        array = array.transpose(1, 0)
    assert array.shape[1] == 4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        # sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color = colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    # now highlight any desired positions; the key of
    # the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos, min_depth],
                                             width=end_pos - start_pos,
                                             height=max_height - min_depth,
                                             edgecolor=color, fill=False))

    ax.set_xlim(-length_padding, array.shape[0] + length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0] + 1, subticks_frequency))
    height_padding = max(abs(min_neg_height) * (height_padding_factor),
                         abs(max_pos_height) * (height_padding_factor))
    ax.set_ylim(min_neg_height - height_padding, max_pos_height + height_padding)


def plot_weights(array,
                 figsize=(20, 2),
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=1.0,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={}):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plot_weights_given_ax(ax=ax, array=array,
                          height_padding_factor=height_padding_factor,
                          length_padding=length_padding,
                          subticks_frequency=subticks_frequency,
                          colors=colors,
                          plot_funcs=plot_funcs,
                          highlight=highlight)
    return fig


# ----- Save-load net part -----


def load_ckp(checkpoint_fpath, model):
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


# ----- Working with DATA part -----


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


def read_fastafile(filename):
    """
    Read a file in FASTA format
    Arguments:
    filename -- string, the name of the sequence file in FASTA format
    Return:
    list of sequences, list of sequence ids
    """
    ids = []
    seqs = []
    try:
        f = open(filename, 'r')
        lines = f.readlines()
        f.close()
    except:
        sys.exit(0)
    seq = []
    for line in lines:
        if line[0] == '>':
            fasta_id = line[1:].rstrip('\n')
            ids.append(line[1:].rstrip('\n'))
            if seq != []: seqs.append("".join(seq))
            seq = []
        else:
            seq.append(line.rstrip('\n').upper())
    if seq != []:
        seqs.append("".join(seq))
    return seqs, ids


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


def get_train_loader(dataset, batch_size):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=3)
    return (train_loader)


def get_3d_feature_matrix(PATH_TO_FASTA_FILE):

    # кол-во нуклеотидов в одном сэмпле (1000 в нашем случае)
    num_nucleotides = SEQ_LENGTH

    # считывание фаста файла с тканью для обучения модели
    # в первой переменной лежит массив всех последовательностей, во второй - их заголовков (с хромосомой и позицией)
    seqs, ids = read_fastafile(PATH_TO_FASTA_FILE)
    print(ids)

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


def get_2d_label_matrix(ids_pos, ids_neg):

    # создаётся массив длинной, равной кол-ву посл-тей во всей тренировочной выборке и размечается 1 и 0
    # данный массив - массив с отметками о принадлежности классу
    y_labels_onehot = []
    for i in range(ids_pos):
        y_labels_onehot.append(LABEL_POS)
    for i in range(ids_neg):
        y_labels_onehot.append(LABEL_NEG)

    # массив становится нумпаевским массивом
    y_labels_onehot = np.array(y_labels_onehot)

    # добавляем массиву вторую размерность, массив становится двумерным с размером {ids_pos + ids_neg}х1
    # (80000 одномерных массивов)
    y_labels_onehot = y_labels_onehot.reshape(-1, 1)

    # преобразовываем каждый из внутренних массивов длины 1 в массив длиной два
    # то есть делаем бинарное отображение категорий (т.е. ответов - ткань или не ткань)
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    y_labels_onehot = onehot_encoder.fit_transform(y_labels_onehot)

    return y_labels_onehot


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


if __name__ == '__main__':

    # --- Preparing tensor from the input file ---

    X_input = get_3d_feature_matrix(PATH_TO_INPUT_FILE)
    X_input = torch.from_numpy(X_input)
    X_input = X_input.float()
    print("Input sequence shape: " + str(X_input.shape))

    X_input = Variable(X_input)
    print("Shape of X_input before permutation: " + str(X_input.shape))
    X_input = X_input.permute(0, 2, 1).contiguous()
    print("Shape of X_input after permutation: " + str(X_input.shape))
    
    # --- Preparing the model ---

    use_gpu = torch.cuda.is_available()
    print("CUDA status: ", use_gpu)

    CNN = Multiple_Input_Model()

    if use_gpu:
        gpu_ids = [0]
        print("Send Model to Cuda")
        CNN = torch.nn.DataParallel(CNN, device_ids=[0])
        X_input = X_input.to(DEVICE_ID)

    # --- Loading net and getting metrics ---

    CNN, start_epoch, valid_loss_min = load_ckp(PATH_TO_SAVE_BEST, CNN)
    #print("model = ", CNN)
    print("start_epoch = ", start_epoch)
    print("valid_loss_min = ", valid_loss_min)

    CNN.eval()

    ig = IntegratedGradients(CNN)
    #ig = DeepLift(CNN)
    attributions, delta = ig.attribute(X_input, target=1, return_convergence_delta=True)

    attributions = attributions.cpu().numpy()
    print("Attributions (ACGT): ")
    print(attributions)
    print("Delta: ")
    print(delta.item())
    print("Attributions shape: ")
    print(attributions.shape)
    print("Z scores:")
    output_vect = output_tensor_to_1d_vector(attributions)
    z_scores = np.array(zscore(output_vect))
    print(z_scores.min())
    print(z_scores.max())
    print("Z scores hist:")
    plt.bar(np.arange(1, 1001), z_scores)
    plt.savefig("bar.png")
    plt.clf()
    plot = sns.distplot(z_scores)
    plot.figure.savefig("dist.png")
    plt.clf()
    print("Image:")
    fig = plot_weights(attributions, subticks_frequency=10, figsize=(5, 5))
    fig.savefig("bio.png")
    plt.clf()

    print("Most remarkable features:")
    most_features = []
    for i in range(len(z_scores)):
        if abs(z_scores[i] > 3):
            most_features.append(i + 1)
    print(most_features)
