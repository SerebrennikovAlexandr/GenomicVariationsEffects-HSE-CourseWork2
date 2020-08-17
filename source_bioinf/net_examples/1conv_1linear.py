import torch
import torchvision
import torchvision.transforms as transforms
import h5py
from sklearn.utils import shuffle
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import optparse
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import sys
from sklearn import metrics
########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].

LABEL_POS = 1
LABEL_NEG = 0
SEQ_LENGTH = 600


"""
PATH_TO_FASTA_FILE_POS = '/mnt/data1/evo_brain_results/selected_data_intersection/chbrain/Chimpanzee_CaudateNucleus.test.fa'
PATH_TO_FASTA_FILE_NEG = '/mnt/data1/evo_brain_results/selected_data_intersection/chbrain_rnd_neg_liftover_from_human/1_to_1_ratio/Human_CaudateNucleus.rnd_sampled_liftover.test.fa'
PATH_TO_VALIDATION_FILE_POS = '/mnt/data1/evo_brain_results/selected_data_intersection/chbrain/Chimpanzee_CaudateNucleus.test.fa'
PATH_TO_VALIDATION_FILE_NEG = '/mnt/data1/evo_brain_results/selected_data_intersection/chbrain_rnd_neg_liftover_from_human/1_to_1_ratio/Human_CaudateNucleus.rnd_sampled_liftover.test.fa'
PATH_TO_SAVE_PREDICTION_SCORES = '/mnt/data1/evo_brain_results/pytorch_model_predictions/prediction.results.txt'
FOLDER_TO_SAVE_MODEL = '/mnt/data1/evo_brain_results/pytorch_model_predictions/'
PATH_TO_SAVE_AUROC_AUPR_SCORES = '/mnt/data1/evo_brain_results/pytorch_model_predictions/prediction.results_auroc_aupr.txt'

"""


parser = optparse.OptionParser()

parser.add_option("-p", "--input_pos_path_file", action="store", type="string", dest="input_pos_path_file",
                  help='Path to input data for training positives')

parser.add_option("-n", "--input_neg_path_file", action="store", type="string", dest="input_neg_path_file",
                  help='Path to input data for training negatives')

parser.add_option("-i", "--input_val_path_pos_file", action="store", type="string", dest="input_val_path_pos_file",
                  help='Path to answers for input data for validation')

parser.add_option("-j", "--input_val_path_neg_file", action="store", type="string", dest="input_val_path_neg_file",
                  help='Path to answers for input data for validation')

parser.add_option("-m", "--output_model_save_path", action="store", type="string", dest="output_model_save_path",
                  help='Folder to save output models')

parser.add_option("-c", "--output_raw_scores_model_save_path", action="store", type="string", dest="output_raw_scores_model_save_path",
                  help='Folder to save output models')

parser.add_option("-r", "--output_roc_scores_model_save_path", action="store", type="string", dest="output_roc_scores_model_save_path",
                  help='Folder to save output models')

(options, args) = parser.parse_args()

PATH_TO_FASTA_FILE_POS = options.input_pos_path_file
PATH_TO_FASTA_FILE_NEG = options.input_neg_path_file
PATH_TO_VALIDATION_FILE_POS = options.input_val_path_pos_file
PATH_TO_VALIDATION_FILE_NEG = options.input_val_path_neg_file
PATH_TO_SAVE_PREDICTION_SCORES = options.output_raw_scores_model_save_path
FOLDER_TO_SAVE_MODEL = options.output_model_save_path
PATH_TO_SAVE_AUROC_AUPR_SCORES = options.output_roc_scores_model_save_path



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
    id = ''
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


###---create np array with zeros---###

def one_hot_encode_along_channel_axis(sequence):
    to_return = np.zeros((len(sequence), 4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return,
                                 sequence=sequence, one_hot_axis=1)
    return to_return


def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis == 0 or one_hot_axis == 1
    if (one_hot_axis == 0):
        assert zeros_array.shape[1] == len(sequence)
    elif (one_hot_axis == 1):
        assert zeros_array.shape[0] == len(sequence)
    # will mutate zeros_array
    for (i, char) in enumerate(sequence):
        if (char == "A" or char == "a"):
            char_idx = 0
        elif (char == "C" or char == "c"):
            char_idx = 1
        elif (char == "G" or char == "g"):
            char_idx = 2
        elif (char == "T" or char == "t"):
            char_idx = 3
        elif (char == "N" or char == "n"):
            continue  # leave that pos as all 0's
        else:
            raise RuntimeError("Unsupported character: " + str(char))
        if (one_hot_axis == 0):
            zeros_array[char_idx, i] = 1
        elif (one_hot_axis == 1):
            zeros_array[i, char_idx] = 1


DEVICE_ID = 1
FILE_TO_SAVE = open(PATH_TO_SAVE_PREDICTION_SCORES, 'w')
FILE_TO_SAVE_SCORES_PER_EPOCH = open(PATH_TO_SAVE_AUROC_AUPR_SCORES,'w')
from scipy.stats import spearmanr, pearsonr
import torch.utils.data as data_utils
import torch.nn as nn




def calculate_dense_size(values_list):
    num_input = 1
    for elem in values_list:
        num_input = num_input * elem
    return num_input


def pearsonr_metric(y_true, y_pred):
    taskwise_pearsonr = [pearsonr(y_true[:, i], y_pred[:, i])[
                             0] for i in range(y_pred.shape[1])]
    return taskwise_pearsonr, np.mean(taskwise_pearsonr)


def pearsonr_metric_modify(y_true, y_pred):
    taskwise_pearsonr = [pearsonr_torch(y_true[:, i], y_pred[:, i])[
                             0] for i in range(y_pred.shape[1])]
    mean_pearson = sum(taskwise_pearsonr) / len(taskwise_pearsonr)
    return taskwise_pearsonr, mean_pearson


def spearmanr_metric_modify(y_true, y_pred):
    taskwise_spearmanr = [spearmanr(y_true[:, i], y_pred[:, i])[
                              0] for i in range(y_pred.shape[1])]
    mean_spearman = sum(taskwise_spearmanr) / len(taskwise_spearmanr)
    return taskwise_spearmanr, mean_spearman


def spearmanr_metric(y_true, y_pred):
    taskwise_spearmanr = [spearmanr(y_true[:, i], y_pred[:, i])[
                              0] for i in range(y_pred.shape[1])]
    return taskwise_spearmanr, np.mean(taskwise_spearmanr)


class Multiple_Input_Model(nn.Module):
    def __init__(self):
        super(Multiple_Input_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=20, stride=1)
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=20, stride=1)
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=20, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=2)
        # self.bn3 = nn.BatchNorm1d(num_features=128)
        # self.bn4 = nn.BatchNorm1d(num_features=256)
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.do = nn.Dropout(p=0.50)
        self.do_conv = nn.Dropout(p=0.5)
        self.l1 = nn.Linear(74240, 2)
        # self.l2 = nn.Linear(256, 2)
        self.softmax_output = nn.Softmax(dim=1)

    def forward(self, x, y):
        ###--Do convolution---###
        x = self.conv_layers1(x)
        y = self.conv_layers1(y)


        # x = self.conv_layers2(x)
        # y = self.conv_layers2(y)
        #
        # x = self.conv_layers3(x)
        # y = self.conv_layers3(y)

        ###---Do ReLu---###
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
        z = self.bn2(z)
        z = self.do(z)
        z = self.relu(z)
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
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    return (loss, optimizer)


num_chanels = 4
seq_length = 600
LEARNING_RATE = 0.15


def get_train_loader(dataset, batch_size, shuffle_or_not = False):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle = shuffle_or_not)
    return (train_loader)


def pearsonr_torch(x, y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y

    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """

    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val.item()


import time
import os
import datetime


def save_tensor_to_csv(tensor_to_save, file_handle, label_idx_text_IDS, epoch_number=0, val_loss=0.0):
    num_rovs = tensor_to_save.size()[0]
    for row_idx in range(0, num_rovs):
        row_with_values = tensor_to_save[row_idx, :]
        string_of_values = [str(i.item()) for i in row_with_values]
        file_handle.write("\t".join(string_of_values) + "\t" + str(epoch_number) + "\t" + str(val_loss) + "\t" + str(label_idx_text_IDS[row_idx]) + "\n")


def trainNet(net, batch_size, n_epochs, learning_rate, dataset_train, val_loader, label_text_id_from_index_dict, use_gpu=False):
    if use_gpu:
        net.to(DEVICE_ID)
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    # Get training data
    train_loader = get_train_loader(dataset_train, batch_size, shuffle_or_not=True)
    n_batches = len(train_loader)

    # Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)

    # scheduler = StepLR(optimizer, step_size=200, gamma=0.99)
    # Time for printing
    training_start_time = time.time()

    # Loop for n_epochs
    for epoch in range(n_epochs):
        # Print Learning Rate
        # print('Epoch:', epoch, 'LR:', scheduler.get_lr())
        print('Epoch:', epoch, 'LR:')
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
            # scheduler.step()
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
            ###---change shape of the input array----###
            inputs_val = inputs_val.permute(0, 2, 1).contiguous()  # batch_size x 4 x 1000
            inputs_val_rev_comp = inputs_val[:, [3, 2, 1, 0], :]
            if use_gpu:
                inputs_val = inputs_val.to(DEVICE_ID)
                inputs_val_rev_comp = inputs_val_rev_comp.to(DEVICE_ID)
                labels_val = labels_val.to(DEVICE_ID)
            ###---Create reverse complement of the sequece---###
            # Forward pass
            val_outputs = net(inputs_val, inputs_val_rev_comp)
            print(val_outputs.size())
            ###---Label text ID works only if batch size is equel to the dataset size---###
            label_text_IDS = [label_text_id_from_index_dict[i] for i in range(0, val_outputs.size()[0])]
            print("label_text_id_from_index_dict", len(label_text_id_from_index_dict))
            print(val_outputs.size())
            print("labels_val", labels_val.size())

            val_loss_size = loss(val_outputs, labels_val)
            val_outputs_cpu = val_outputs.cpu().data.numpy()
            labels_cpu = labels_val.cpu().data.numpy()
            concatenated_two_tensors = torch.cat((val_outputs, labels_val), dim=1)



            total_val_loss += val_loss_size.item()
            true_labels = decode_one_hot_class_labels(labels_val)
            predicted_lables_for_pos_class = get_scores_pos_class_labels(val_outputs)

            # print(true_labels)
            # print(val_outputs)
            fpr, tpr, thresholds = metrics.roc_curve(true_labels, predicted_lables_for_pos_class)
            precision, recall, _ = metrics.precision_recall_curve(true_labels, predicted_lables_for_pos_class)
            auroc_score = metrics.auc(fpr, tpr)
            aupr = metrics.auc(recall,precision)
            FILE_TO_SAVE_SCORES_PER_EPOCH.write(str(epoch) + "\t" + str(auroc_score) + "\t" + str(aupr) + "\n")

            print("aupr:", aupr, " auroc_score:", auroc_score)
            save_tensor_to_csv(concatenated_two_tensors, FILE_TO_SAVE, label_idx_text_IDS = label_text_IDS, epoch_number=epoch,
                               val_loss=val_loss_size.item())

            # save_tensor_to_csv(concatenated_two_tensors, FILE_TO_SAVE, epoch_number=epoch,
            #                    val_loss=val_loss_size.item())


        acc = total_val_loss
        ###----Save model every epoch---###
        FILENAME_SAVE_MODEL = FOLDER_TO_SAVE_MODEL + "/" + os.path.join(
            'cnnT1_epoch_{}_loss_{}_{}.t7'.format(epoch + 1, acc, datetime.datetime.now().strftime("%b_%d_%H:%M:%S")))
        # torch.save(net.state_dict(), FILENAME_SAVE_MODEL)

        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

num_nucleotides = SEQ_LENGTH

seqs_pos, ids_pos = read_fastafile(PATH_TO_FASTA_FILE_POS)
n_rows_pos = len(ids_pos)
X_data_matrix_pos = np.zeros((n_rows_pos, num_nucleotides, 4))
##----Read fasta_file----###
for i, fasta_seq in enumerate(seqs_pos):
    one_hot_encoded = one_hot_encode_along_channel_axis(fasta_seq)
    X_data_matrix_pos[i, :, :] = one_hot_encoded
X_data_matrix_pos[np.isnan(X_data_matrix_pos)] = 0

seqs_neg, ids_neg = read_fastafile(PATH_TO_FASTA_FILE_NEG)
n_rows_neg = len(ids_neg)
X_data_matrix_neg = np.zeros((n_rows_neg, num_nucleotides, 4))
##----Read fasta_file----###
for i, fasta_seq in enumerate(seqs_neg):
    one_hot_encoded = one_hot_encode_along_channel_axis(fasta_seq)
    X_data_matrix_neg[i, :, :] = one_hot_encoded
X_data_matrix_neg[np.isnan(X_data_matrix_neg)] = 0

X_train_matrix = np.concatenate((X_data_matrix_pos, X_data_matrix_neg))

###---create vector with pos and neg---###

y_labels_train_onehot = []

for i in ids_pos:
    y_labels_train_onehot.append(LABEL_POS)

for i in ids_neg:
    y_labels_train_onehot.append(LABEL_NEG)

y_labels_train_onehot = np.array(y_labels_train_onehot)
y_labels_train_onehot = y_labels_train_onehot.reshape(-1,1)
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
y_labels_train_onehot = onehot_encoder.fit_transform(y_labels_train_onehot)


###---oad validation pos and neg data---###

num_nucleotides = SEQ_LENGTH

seqs_pos_val, ids_pos_val = read_fastafile(PATH_TO_VALIDATION_FILE_POS)
n_rows_pos_val = len(ids_pos_val)
X_data_matrix_pos_val = np.zeros((n_rows_pos_val, num_nucleotides, 4))
##----Read fasta_file----###
for i, fasta_seq in enumerate(seqs_pos_val):
    one_hot_encoded = one_hot_encode_along_channel_axis(fasta_seq)
    X_data_matrix_pos_val[i, :, :] = one_hot_encoded
X_data_matrix_pos_val[np.isnan(X_data_matrix_pos_val)] = 0

seqs_neg_val, ids_neg_val = read_fastafile(PATH_TO_VALIDATION_FILE_NEG)
n_rows_neg_val = len(ids_neg_val)
X_data_matrix_neg_val = np.zeros((n_rows_neg_val, num_nucleotides, 4))
##----Read fasta_file----###
for i, fasta_seq in enumerate(seqs_neg_val):
    one_hot_encoded = one_hot_encode_along_channel_axis(fasta_seq)
    X_data_matrix_neg_val[i, :, :] = one_hot_encoded
X_data_matrix_neg_val[np.isnan(X_data_matrix_neg_val)] = 0

X_val_matrix = np.concatenate((X_data_matrix_pos_val, X_data_matrix_neg_val))

###---create vector with pos and neg---###

y_labels_val = []

for i in ids_pos_val:
    y_labels_val.append(LABEL_POS)

for i in ids_neg_val:
    y_labels_val.append(LABEL_NEG)


y_labels_val = np.array(y_labels_val)
y_labels_val = y_labels_val.reshape(-1,1)
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
y_labels_val_onehot = onehot_encoder.fit_transform(y_labels_val)


region_IDs = ids_pos_val + ids_neg_val


region_ID_2_ylabel = dict()
for i, label_id in enumerate(region_IDs):
    region_ID_2_ylabel[i] = label_id

X_train = torch.from_numpy(X_train_matrix)
X_train = X_train.float()
Y_train = torch.from_numpy(y_labels_train_onehot)
Y_train = Y_train.float()



X_train_val = torch.from_numpy(X_val_matrix)
X_train_val = X_train_val.float()
Y_train_val = torch.from_numpy(y_labels_val_onehot)
Y_train_val = Y_train_val.float()
print(Y_train_val.size())


print("Validation set size:", X_train_val.size())
print("Training set size:", X_train.size())
use_gpu = torch.cuda.is_available()

print("Cuda status", use_gpu)
BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = len(y_labels_val_onehot)


CNN = Multiple_Input_Model()




if use_gpu:
    gpu_ids = [1, 2, 3]
    print("Send Model to Cuda")
    CNN = torch.nn.DataParallel(CNN, device_ids=[1, 2, 3])
    train_data = data_utils.TensorDataset(X_train, Y_train)
    val_data = data_utils.TensorDataset(X_train_val, Y_train_val)
    val_loader = get_train_loader(dataset = val_data, batch_size=VALIDATION_BATCH_SIZE, shuffle_or_not= False)
else:
    train_data = data_utils.TensorDataset(X_train, Y_train)
    val_data = data_utils.TensorDataset(X_train_val, Y_train_val)
    val_loader = get_train_loader(dataset = val_data, batch_size=VALIDATION_BATCH_SIZE, shuffle_or_not= False)
trainNet(CNN, batch_size=BATCH_SIZE,  n_epochs=200, learning_rate=LEARNING_RATE, dataset_train=train_data,
         val_loader=val_loader, label_text_id_from_index_dict = region_ID_2_ylabel, use_gpu=use_gpu)
