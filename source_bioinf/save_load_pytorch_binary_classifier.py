import torch
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
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


LOAD_NET = False
DEVICE_ID = 0


LABEL_POS = 1
LABEL_NEG = 0
SEQ_LENGTH = 1000
LEARNING_RATE = 0.15
BATCH_SIZE = 32


PATH_TO_FASTA_FILE_POS = r"../H3K4me3_learn.fasta"
PATH_TO_FASTA_FILE_NEG = r"../H3K4me3_random_learn.fasta"
PATH_TO_VALIDATION_FILE_POS = r"../H3K4me3_test.fasta"
PATH_TO_VALIDATION_FILE_NEG = r"../H3K4me3_random_test.fasta"
PATH_TO_SAVE_PREDICTION_SCORES = r"../model_results/prediction_results.txt"
PATH_TO_SAVE_AUROC_AUPR_SCORES = r"../model_results/results_auroc_aupr.txt"
FOLDER_TO_SAVE_CHECKPOINT = r"../model_results/saved_model/"
PATH_TO_SAVE_BEST = r"../model_results/best_model/best_checkpoint.pt"


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


FILE_TO_SAVE = open(PATH_TO_SAVE_PREDICTION_SCORES, 'w')
FILE_TO_SAVE_SCORES_PER_EPOCH = open(PATH_TO_SAVE_AUROC_AUPR_SCORES,'w')


class Multiple_Input_Model(nn.Module):
    def __init__(self):
        super(Multiple_Input_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=20, stride=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=20, stride=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=20, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.bn4 = nn.BatchNorm1d(num_features=256)
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.do = nn.Dropout(p=0.50)
        self.do_conv = nn.Dropout(p=0.5)
        self.l1 = nn.Linear(241152, 256)
        self.l2 = nn.Linear(256, 2)
        self.softmax_output = nn.Softmax(dim=1)
        self.flatten = lambda x: x.view(-1)

    def forward(self, x, y):
        # --Do convolution--- #
        x = self.conv_layers1(x)
        y = self.conv_layers1(y)

        x = self.conv_layers2(x)
        y = self.conv_layers2(y)

        x = self.conv_layers3(x)
        y = self.conv_layers3(y)

        # ---Do ReLu--- #
        x_max_pool = self.mp(x)
        y_max_pool = self.mp(y)
        x_avg_pool = self.avg_pool(x)
        y_avg_pool = self.avg_pool(y)
        z_max_pool = torch.cat((x_max_pool, y_max_pool), 1)
        z_avg_pool = torch.cat((x_avg_pool, y_avg_pool), 1)
        z = torch.cat((z_max_pool, z_avg_pool), 1)
        N, _, _ = z.size()
        z = z.view(N, -1)
        # z = self.bn1(z)
        # print(z.size())
        z = self.l1(z)
        z = self.bn4(z)
        z = self.relu(z)
        z = self.do(z)
        z = self.l2(z)
        z = self.softmax_output(z)
        return z

    def conv_layers1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.do_conv(x)
        return x

    def conv_layers2(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.do_conv(x)
        return x

    def conv_layers3(self, x):
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.do_conv(x)
        return x


# criterion = torch.nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def createLossAndOptimizer(net, learning_rate=0.1):
    # Loss function
    loss = nn.BCELoss()
    # Optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)
    return (loss, optimizer)


def get_train_loader(dataset, batch_size):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=3)
    return (train_loader)


def save_tensor_to_csv(tensor_to_save, file_handle, epoch_number=0, val_loss=0.0):
    num_rovs = tensor_to_save.size()[0]
    for row_idx in range(0, num_rovs):
        row_with_values = tensor_to_save[row_idx, :]
        string_of_values = [str(i.item()) for i in row_with_values]
        file_handle.write("\t".join(string_of_values) + "\t" + str(epoch_number) + "\t" + str(val_loss) + "\n")


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


def trainNet(net, batch_size, n_epochs, learning_rate, dataset_train, val_loader, valid_loss_min_input, use_gpu=False, start_epochs=0):

    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    if use_gpu:
        net.to(DEVICE_ID)
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    # Get training data
    train_loader = get_train_loader(dataset_train, batch_size)
    n_batches = len(train_loader)

    # Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)

    scheduler = StepLR(optimizer, step_size=200, gamma=0.99)
    # Time for printing
    training_start_time = time.time()

    # Loop for n_epochs
    for epoch in range(start_epochs, n_epochs):
        # Print Learning Rate
        print('Epoch:', epoch, 'LR:', scheduler.get_lr())
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        for i, data in enumerate(train_loader, 0):
            # Get inputs
            inputs, labels = data
            # Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            inputs = inputs.permute(0, 2, 1).contiguous()  # batch_size x 4 x 1000
            inputs_rev_comp = inputs[:, [3, 2, 1, 0], :]
            if use_gpu:
                if i % 1000 == 0:
                    print("Sending tensor to CUDA")
                inputs = inputs.to(DEVICE_ID)
                inputs_rev_comp = inputs_rev_comp.to(DEVICE_ID)
                labels = labels.to(DEVICE_ID)
            # Set the parameter gradients to zero
            optimizer.zero_grad()
            # Forward pass, backward pass, optimize
            outputs = net(inputs, inputs_rev_comp)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            scheduler.step()
            # Print statistics
            running_loss += loss_size.item()
            total_train_loss += loss_size.item()
            # Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time))
                # Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0

        # concatenated_tensor_list = []
        for inputs_val, labels_val in val_loader:
            # Wrap tensors in Variables
            inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)
            # change shape of the input array
            inputs_val = inputs_val.permute(0, 2, 1).contiguous()  # batch_size x 4 x 1000
            inputs_val_rev_comp = inputs_val[:, [3, 2, 1, 0], :]
            if use_gpu:
                inputs_val = inputs_val.to(DEVICE_ID)
                inputs_val_rev_comp = inputs_val_rev_comp.to(DEVICE_ID)
                labels_val = labels_val.to(DEVICE_ID)

            # Create reverse complement of the sequece

            # Forward pass
            val_outputs = net(inputs_val, inputs_val_rev_comp)
            val_loss_size = loss(val_outputs, labels_val)
            #val_outputs_cpu = val_outputs.cpu().data.numpy()
            #labels_cpu = labels_val.cpu().data.numpy()
            concatenated_two_tensors = torch.cat((val_outputs, labels_val), dim=1)
            total_val_loss += val_loss_size.item()
            true_labels = decode_one_hot_class_labels(labels_val)
            predicted_lables_for_pos_class = get_scores_pos_class_labels(val_outputs)

            fpr, tpr, thresholds = metrics.roc_curve(true_labels, predicted_lables_for_pos_class)
            precision, recall, _ = metrics.precision_recall_curve(true_labels, predicted_lables_for_pos_class)
            auroc_score = metrics.auc(fpr, tpr)
            aupr = metrics.auc(recall,precision)
            FILE_TO_SAVE_SCORES_PER_EPOCH.write(str(epoch) + "\t" + str(auroc_score) + "\t" + str(aupr) + "\n")

            print("aupr:", aupr, " auroc_score:", auroc_score)
            save_tensor_to_csv(concatenated_two_tensors, FILE_TO_SAVE, epoch_number=epoch,
                               val_loss=val_loss_size.item())

        acc = total_val_loss

        # --- Save model every epoch --- #

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': acc,
            'state_dict': net.state_dict()
        }

        # save checkpoint
        FILENAME_SAVE_CHECKPOINT = FOLDER_TO_SAVE_CHECKPOINT + os.path.join(
            'cnnT1_epoch_{}_loss_{}_{}.t7'.format(epoch + 1, acc, datetime.datetime.now().strftime("%b_%d_%H_%M_%S")))

        save_ckp(checkpoint, False, FILENAME_SAVE_CHECKPOINT, PATH_TO_SAVE_BEST)

        if acc <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, acc))
            # save checkpoint as best model
            save_ckp(checkpoint, True, FILENAME_SAVE_CHECKPOINT, PATH_TO_SAVE_BEST)
            valid_loss_min = acc

        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


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

    # созлание двумерного бинарного массива меток о принадлежности к ткани для обучения
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

    # созлание двумерного бинарного массива меток о принадлежности к ткани для обучения
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
    val_loader = get_train_loader(val_data, 32)

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

        '''
        trainNet(CNN, batch_size=BATCH_SIZE, n_epochs=30, learning_rate=LEARNING_RATE, dataset_train=train_data,
                 val_loader=val_loader, valid_loss_min_input=valid_loss_min, use_gpu=use_gpu, start_epochs=start_epoch)
        '''

    else:
        trainNet(CNN, batch_size=BATCH_SIZE, n_epochs=30, learning_rate=LEARNING_RATE, dataset_train=train_data,
                 val_loader=val_loader, valid_loss_min_input=np.Inf, use_gpu=use_gpu)
