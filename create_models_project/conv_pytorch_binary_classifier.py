import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
import sys
from sklearn import metrics
import torch.utils.data as data_utils
import torch.nn as nn
import time
import os
import datetime


########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].


# ----- Settings part -----


LOAD_NET = True
DEVICE_ID = 0


LABEL_POS = 1
LABEL_NEG = 0
SEQ_LENGTH = 1000
LEARNING_RATE = 0.15
BATCH_SIZE = 128
N_EPOCHS = 30


PATH_TO_FASTA_FILE_POS = r"gen_samples_data/H3K4me3/train_data/H3K4me3_learn.fasta"
PATH_TO_FASTA_FILE_NEG = r"gen_samples_data/H3K4me3/train_data/H3K4me3_random_learn.fasta"
PATH_TO_VALIDATION_FILE_POS = r"gen_samples_data/H3K4me3/train_data/H3K4me3_test.fasta"
PATH_TO_VALIDATION_FILE_NEG = r"gen_samples_data/H3K4me3/train_data/H3K4me3_random_test.fasta"
#PATH_TO_SAVE_PREDICTION_SCORES = r"../model_results/prediction_results.txt"
#PATH_TO_SAVE_AUROC_AUPR_SCORES = r"../model_results/results_auroc_aupr.txt"
FOLDER_TO_SAVE_CHECKPOINT = r"../model_results/H3K4me3/saved_model/"
PATH_TO_SAVE_BEST = r"../model_results/H3K4me3/best_model/best_checkpoint.pt"


# ----- Save-load net part -----


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        torch.save(state, best_fpath)


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


# ----- Net-defining part -----


class Multiple_Input_Model(nn.Module):
    def __init__(self):
        super(Multiple_Input_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=300, kernel_size=8, stride=1)
        self.conv2 = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=5, stride=1)
        self.conv3 = nn.Conv1d(in_channels=300, out_channels=150, kernel_size=4, stride=1)
        self.conv4 = nn.Conv1d(in_channels=150, out_channels=150, kernel_size=4, stride=1)
        self.conv5 = nn.Conv1d(in_channels=150, out_channels=300, kernel_size=4, stride=1)
        self.conv6 = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=4, stride=1)
        self.conv7 = nn.Conv1d(in_channels=300, out_channels=256, kernel_size=4, stride=1)
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=4, stride=1)
        self.relu = nn.ELU(inplace=True)
        self.bn1 = nn.BatchNorm1d(num_features=300)
        self.bn2 = nn.BatchNorm1d(num_features=300)
        self.bn3 = nn.BatchNorm1d(num_features=150)
        self.bn4 = nn.BatchNorm1d(num_features=150)
        self.bn5 = nn.BatchNorm1d(num_features=300)
        self.bn6 = nn.BatchNorm1d(num_features=300)
        self.bn7 = nn.BatchNorm1d(num_features=256)
        self.bn8 = nn.BatchNorm1d(num_features=256)
        self.bn_fc_1 = nn.BatchNorm1d(num_features=256)
        self.bn_fc_2 = nn.BatchNorm1d(num_features=128)
        self.bn_fc_3 = nn.BatchNorm1d(num_features=2)

        self.mp_1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.mp_2 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.mp_3 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.mp_4 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.mp_5 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.mp_6 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.mp_7 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.mp_8 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.do = nn.Dropout(p=0.30)
        self.do_conv = nn.Dropout(p=0.40)
        self.l1 = nn.Linear(296400, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 2)
        self.softmax_output = nn.Sigmoid()

    def forward(self, x):
        # -- Do convolution -- #
        x = self.conv_layers1(x)
        x = self.conv_layers2(x)
        # x = self.conv_layers3(x)
        # x = self.conv_layers4(x)
        # x = self.conv_layers5(x)
        # x = self.conv_layers6(x)
        # x = self.conv_layers7(x)
        # x = self.conv_layers8(x)

        # -- Do ReLu -- #
        N, _, _ = x.size()
        x = x.view(N, -1)

        x = self.linear_layer1(x)
        x = self.linear_layer2(x)
        x = self.linear_layer3(x)
        x = self.softmax_output(x)
        return x

    def conv_layers1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.do_conv(x)
        # x = self.mp_1(x)
        return x

    def conv_layers2(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.do_conv(x)
        x = self.mp_2(x)
        return x

    def conv_layers3(self, x):
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.do_conv(x)
        # x = self.mp_3(x)
        return x

    def conv_layers4(self, x):
        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn4(x)
        # x = self.do_conv(x)
        x = self.mp_4(x)
        return x

    def conv_layers5(self, x):
        x = self.conv5(x)
        x = self.relu(x)
        x = self.bn5(x)
        x = self.do_conv(x)
        # x = self.mp_5(x)
        return x

    def conv_layers6(self, x):
        x = self.conv6(x)
        x = self.relu(x)
        x = self.bn6(x)
        # x = self.do_conv(x)
        x = self.mp_6(x)
        return x


    def conv_layers7(self, x):
        x = self.conv7(x)
        x = self.relu(x)
        x = self.bn7(x)
        x = self.do_conv(x)
        # x = self.mp_7(x)
        return x

    def conv_layers8(self, x):
        x = self.conv8(x)
        x = self.relu(x)
        x = self.bn8(x)
        # x = self.do_conv(x)
        x = self.mp_8(x)
        return x

    def linear_layer1(self, x):
        x = self.l1(x)
        x = self.bn_fc_1(x)
        x = self.relu(x)
        x = self.do(x)
        return x

    def linear_layer2(self, x):
        x = self.l2(x)
        x = self.bn_fc_2(x)
        x = self.relu(x)
        # x = self.do(x)
        return x

    def linear_layer3(self, x):
        x = self.l3(x)
        x = self.relu(x)
        # x = self.do(x)
        return x


# ----- Working with DATA part -----


def decode_one_hot_class_labels(tensor_with_true_scores):
    # принимает тензор размера Nx2 (2 значения в каждой строке)
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

    # возвращает вектор Numpy, раскодированный по второму столбцу тензора (если 1.0 - то 1, иначе 0)
    return vector_with_labels


def get_scores_pos_class_labels(tensor_with_true_scores):
    # принимает тензор размера Nx2 (2 значения в каждой строке)
    vector_with_labels = []
    pos_idx = 1
    neg_idx = 0
    for row_idx in range(0, tensor_with_true_scores.size()[0]):
        vector_with_labels.append(tensor_with_true_scores[row_idx, pos_idx].item())

    # возвращает вектор Numpy, скопированный из второго столбца тензора
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


def get_train_loader(dataset, batch_size, shuffle_or_not=False):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_or_not)
    return train_loader


def get_3d_feature_matrix(PATH_TO_FASTA_FILE):

    # кол-во нуклеотидов в одном сэмпле (1000 в нашем случае)
    num_nucleotides = SEQ_LENGTH

    # считывание фаста файла с тканью для обучения модели
    # в первой переменной лежит массив всех последовательностей, во второй - их заголовков (с хромосомой и позицией)
    seqs, ids = read_fastafile(PATH_TO_FASTA_FILE)

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


# ----- Train-defining part -----


def train(net, batch_size, epoch, dataset_train, criterion, use_gpu):
    # Переключение сети в режим тренировки
    net.train()
    # Данный параметр вычисляется, а не передаётся
    learning_rate = 0.01 * (0.97 ** epoch)
    print("learning_rate=", learning_rate)

    # Сделать вывод по новой эпохе

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.98, weight_decay=5e-4)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    n_batches = len(train_loader)

    print_every = n_batches // 10
    for batch_idx, (train_seqs, train_targets) in enumerate(train_loader):
        if (batch_idx + 1) % (print_every + 1) == 0:
            print("Processed ", int(100 * (batch_idx + 1) / n_batches), " of the data in EPOCH")

        train_seqs, train_targets = Variable(train_seqs), Variable(train_targets)

        if use_gpu:
            train_seqs, train_targets = train_seqs.to(DEVICE_ID), train_targets.to(DEVICE_ID)

        train_seqs = train_seqs.permute(0, 2, 1).contiguous()  # batch_size x 4 x 1000
        optimizer.zero_grad()
        train_outputs = net(train_seqs)
        loss = criterion(train_outputs, train_targets)
        loss.backward()
        optimizer.step()


def valid(net, valid_loader, criterion):
    net.eval()
    y_pred = []
    y_true = []
    y_loss = 0
    for batch_idx, (valid_seqs, valid_targets) in enumerate(valid_loader):
        if use_gpu:
            valid_seqs, valid_targets = valid_seqs.to(DEVICE_ID), valid_targets.to(DEVICE_ID)

        valid_seqs, valid_targets = Variable(valid_seqs, volatile=True), Variable(valid_targets)
        valid_seqs = valid_seqs.permute(0, 2, 1).contiguous()
        valid_outputs = net(valid_seqs)

        y_true_batch = decode_one_hot_class_labels(valid_targets)
        y_pred_batch = get_scores_pos_class_labels(valid_outputs)

        for elem in y_true_batch:
            y_true.append(elem)
        for elem in y_pred_batch:
            y_pred.append(elem)

        loss = criterion(valid_outputs, valid_targets)
        y_loss += loss.data.cpu().numpy()
    return np.array(y_true), np.array(y_pred), y_loss


def trainNet(net, batch_size, n_epochs, dataset_train, val_loader, valid_loss_min_input, use_gpu=False, start_epochs=0):
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("=" * 30)

    epoch = max(0, start_epochs)
    criterion = nn.BCELoss()

    while epoch < n_epochs:
        print('\nEpoch: %d' % epoch)

        train(net, batch_size=batch_size, epoch=epoch, dataset_train=dataset_train,
              criterion=criterion, use_gpu=use_gpu)

        true, pred, y_loss = valid(net, val_loader, criterion)

        fpr, tpr, thresholds = metrics.roc_curve(true, pred)
        precision, recall, _ = metrics.precision_recall_curve(true, pred)
        auroc_score = metrics.auc(fpr, tpr)
        aupr = metrics.auc(recall, precision)

        print(auroc_score, " ", aupr) #заменить

        acc = y_loss

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': acc,
            'state_dict': net.state_dict()
        }

        # save checkpoint
        FILENAME_SAVE_CHECKPOINT = FOLDER_TO_SAVE_CHECKPOINT + os.path.join(
            'cnnT1_epoch_{}_loss_{}_{}.t7'.format(epoch + 1, acc, datetime.datetime.now().strftime("%b_%d_%H_%M_%S")))

        if acc <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, acc))
            # save checkpoint as best model
            save_ckp(checkpoint, True, FILENAME_SAVE_CHECKPOINT, PATH_TO_SAVE_BEST)
            valid_loss_min = acc
        else:
            save_ckp(checkpoint, False, FILENAME_SAVE_CHECKPOINT, PATH_TO_SAVE_BEST)

        epoch += 1

    print("=" * 30)
    print("Training finished")


# ----- Main part -----


if __name__ == '__main__':

    # --- Read pos-learn fasta file ---

    # создание трёхмерного массива в бинарном формате с тканью для обучения:
    # x: 20000
    # y: 1000 в нашем случае
    # z: 4 - ACGT
    X_data_matrix_pos = get_3d_feature_matrix(PATH_TO_FASTA_FILE_POS)

    # --- Read neg-learn fasta_file ---

    # создание трёхмерного массива в бинарном формате с random sample для обучения:
    # x: 60000
    # y: 1000 в нашем случае
    # z: 4 - ACGT
    X_data_matrix_neg = get_3d_feature_matrix(PATH_TO_FASTA_FILE_NEG)

    # конкатенация массивов pos и neg для обучения
    # итоговый размер матрицы для тренировки: 80000х1000х4
    X_train_matrix = np.concatenate((X_data_matrix_pos, X_data_matrix_neg))

    # --- Create vector with pos and neg for training ---

    # создание двумерного бинарного массива меток о принадлежности к ткани для обучения
    y_labels_train_onehot = get_2d_label_matrix(X_data_matrix_pos.shape[0], X_data_matrix_neg.shape[0])

    # --- Read pos-validate fasta file ---

    # создание трёхмерного массива в бинарном формате с тканью для теста:
    # x: 3000
    # y: 1000 в нашем случае
    # z: 4 - ACGT
    X_data_matrix_pos_val = get_3d_feature_matrix(PATH_TO_VALIDATION_FILE_POS)

    # --- Read neg-validate fasta file ---

    # создание трёхмерного массива в бинарном формате с random sample для теста:
    # x: 9000
    # y: 1000 в нашем случае
    # z: 4 - ACGT
    X_data_matrix_neg_val = get_3d_feature_matrix(PATH_TO_VALIDATION_FILE_NEG)

    # конкатенация массивов pos и neg для теста
    # итоговый размер матрицы для тренировки: 12000х1000х4
    X_val_matrix = np.concatenate((X_data_matrix_pos_val, X_data_matrix_neg_val))

    # --- Create vector with pos and neg for validation ---

    # создание двумерного бинарного массива меток о принадлежности к ткани для обучения
    y_labels_val_onehot = get_2d_label_matrix(X_data_matrix_pos_val.shape[0], X_data_matrix_neg_val.shape[0])

    # --- Shuffling ---

    # X_train_matrix, X_val_matrix, y_labels_train_onehot, y_labels_val_onehot = shuffle(X_train_matrix, X_val_matrix, y_labels_train_onehot, y_labels_val_onehot, random_state=0)

    # --- Moving numpy arrays to torch tensors ---

    X_train = torch.from_numpy(X_train_matrix)
    X_train = X_train.float()
    Y_train = torch.from_numpy(y_labels_train_onehot)
    Y_train = Y_train.float()

    X_val = torch.from_numpy(X_val_matrix)
    X_val = X_val.float()
    Y_val = torch.from_numpy(y_labels_val_onehot)
    Y_val = Y_val.float()

    print("Validation set size:", X_val.size())
    print("Training set size:", X_train.size())
    use_gpu = torch.cuda.is_available()

    print("Cuda status", use_gpu)

    CNN = Multiple_Input_Model()

    print(CNN)

    train_data = data_utils.TensorDataset(X_train, Y_train)
    val_data = data_utils.TensorDataset(X_val, Y_val)
    val_loader = get_train_loader(val_data, batch_size=BATCH_SIZE, shuffle_or_not=False)

    if use_gpu:
        gpu_ids = [0]
        print("Send Model to Cuda")
        CNN = torch.nn.DataParallel(CNN, device_ids=[0])

    # --- loading saved model, training and analyzing it --- #
    if LOAD_NET:
        CNN, start_epoch, valid_loss_min = load_ckp(PATH_TO_SAVE_BEST, CNN)
        print("model = ", CNN)
        print("start_epoch = ", start_epoch)
        print("valid_loss_min = ", valid_loss_min)

        trainNet(CNN, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, dataset_train=train_data,
                 val_loader=val_loader, valid_loss_min_input=valid_loss_min, use_gpu=use_gpu, start_epochs=start_epoch)

    else:
        trainNet(CNN, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, dataset_train=train_data,
                 val_loader=val_loader, valid_loss_min_input=np.Inf, use_gpu=use_gpu)
