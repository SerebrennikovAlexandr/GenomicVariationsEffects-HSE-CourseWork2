import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import optparse
from sklearn.preprocessing import OneHotEncoder
import sys
from sklearn import metrics
import datetime
import os
import torch.utils.data as data_utils
import torch.nn as nn

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].


LABEL_POS = 1
LABEL_NEG = 0
SEQ_LENGTH = 600
LEARNING_RATE = 0.0001
gpu_ids = [1, 2, 3]
DEVICE_ID = 1
BATCH_SIZE = 128


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

parser.add_option("-c", "--output_raw_scores_model_save_path", action="store", type="string",
                  dest="output_raw_scores_model_save_path",
                  help='Folder to save output models')

parser.add_option("-r", "--output_roc_scores_model_save_path", action="store", type="string",
                  dest="output_roc_scores_model_save_path",
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


FILE_TO_SAVE = open(PATH_TO_SAVE_PREDICTION_SCORES, 'w')
FILE_TO_SAVE_SCORES_PER_EPOCH = open(PATH_TO_SAVE_AUROC_AUPR_SCORES, 'w')


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
        self.l1 = nn.Linear(135680, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, 2)
        self.softmax_output = nn.Sigmoid()


    def forward(self, x):
        ###--Do convolution---###
        x = self.conv_layers1(x)
        x = self.conv_layers2(x)
        # x = self.conv_layers3(x)
        # x = self.conv_layers4(x)
        # x = self.conv_layers5(x)
        # x = self.conv_layers6(x)
        # x = self.conv_layers7(x)
        # x = self.conv_layers8(x)

        ###---Do ReLu---###
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


def get_train_loader(dataset, batch_size, shuffle_or_not=False):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_or_not)
    return (train_loader)


def train(epoch, train_data, batch_size):
    net.train()
    mylr = 0.01 * (0.97 ** epoch)
    # mylr = 0.02 * (0.7 ** epoch)
    print("Learning rate", mylr)
    optimizer = optim.SGD(net.parameters(), lr=mylr, momentum=0.98, weight_decay=5e-4)
    #
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    n_batches = len(train_loader)
    print_every = n_batches // 10
    for batch_idx, (train_seqs, train_targets) in enumerate(train_loader):
        if (batch_idx + 1) % (print_every + 1) == 0:
            print("Processed ", int(100 * (batch_idx + 1) / n_batches), " of the data in EPOCH")
        train_seqs, train_targets = Variable(train_seqs), Variable(train_targets)
        # train_seqs_rev_comp = train_seqs[:, [3, 2, 1, 0], :]
        if use_gpu:
            train_seqs, train_targets = train_seqs.to(DEVICE_ID), train_targets.to(DEVICE_ID)
        train_seqs = train_seqs.permute(0, 2, 1).contiguous()  # batch_size x 4 x 1000
        optimizer.zero_grad()
        train_outputs = net(train_seqs)
        loss = criterion(train_outputs, train_targets)
        loss.backward()
        optimizer.step()

    FILENAME_SAVE_MODEL = FOLDER_TO_SAVE_MODEL + "/" + os.path.join(
        'cnnT1_epoch_{}_{}.t7'.format(epoch + 1, datetime.datetime.now().strftime("%b_%d_%H:%M:%S")))
    torch.save(net.state_dict(), FILENAME_SAVE_MODEL)



def valid(validloader, batch_size):
    net.eval()
    y_pred = []
    y_true = []
    y_loss = 0
    for batch_idx, (valid_seqs, valid_targets) in enumerate(validloader):
        if use_gpu:
            # valid_seqs, valid_targets = valid_seqs.cuda(), valid_targets.cuda()
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






num_nucleotides = SEQ_LENGTH

seqs_pos, ids_pos = read_fastafile(PATH_TO_FASTA_FILE_POS)
n_rows_pos = len(ids_pos)
X_data_matrix_pos = np.zeros((n_rows_pos, num_nucleotides, 4))
##----Read fasta_file----###
for i, fasta_seq in enumerate(seqs_pos):
    one_hot_encoded = one_hot_encode_along_channel_axis(fasta_seq)
    X_data_matrix_pos[i, :, :] = one_hot_encoded
# X_data_matrix_pos[np.isnan(X_data_matrix_pos)] = 0

seqs_neg, ids_neg = read_fastafile(PATH_TO_FASTA_FILE_NEG)
n_rows_neg = len(ids_neg)
X_data_matrix_neg = np.zeros((n_rows_neg, num_nucleotides, 4))
##----Read fasta_file----###
for i, fasta_seq in enumerate(seqs_neg):
    one_hot_encoded = one_hot_encode_along_channel_axis(fasta_seq)
    X_data_matrix_neg[i, :, :] = one_hot_encoded
# X_data_matrix_neg[np.isnan(X_data_matrix_neg)] = 0

X_train_matrix = np.concatenate((X_data_matrix_pos, X_data_matrix_neg))
print("Concat matrix shape", X_train_matrix.shape)
###---create vector with pos and neg---###

y_labels_train_onehot = []

for i in ids_pos:
    y_labels_train_onehot.append(LABEL_POS)

for i in ids_neg:
    y_labels_train_onehot.append(LABEL_NEG)

onehot_encoder = OneHotEncoder(sparse=False, categories='auto')

y_labels_train_onehot = np.array(y_labels_train_onehot)

y_labels_train_onehot = y_labels_train_onehot.reshape(-1, 1)
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
y_labels_val = y_labels_val.reshape(-1, 1)
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


print("Validation set size:", X_train_val.size())
print("Training set size:", X_train.size())
use_gpu = torch.cuda.is_available()
print("Cuda status", use_gpu)

net = Multiple_Input_Model()

# VALIDATION_BATCH_SIZE= len(y_labels_val_onehot)
VALIDATION_BATCH_SIZE = BATCH_SIZE
if use_gpu:
    gpu_ids = [1, 2, 3]
    print("Send Model to Cuda")
    net = torch.nn.DataParallel(net, device_ids=[1, 2, 3])
    net.to(DEVICE_ID)
    train_data = data_utils.TensorDataset(X_train, Y_train)
    val_data = data_utils.TensorDataset(X_train_val, Y_train_val)
    val_loader = get_train_loader(dataset=val_data, batch_size=VALIDATION_BATCH_SIZE, shuffle_or_not=False)
else:
    train_data = data_utils.TensorDataset(X_train, Y_train)
    val_data = data_utils.TensorDataset(X_train_val, Y_train_val)
    val_loader = get_train_loader(dataset=val_data, batch_size=VALIDATION_BATCH_SIZE, shuffle_or_not=False)


# trainNet(CNN, batch_size=BATCH_SIZE, n_epochs=200, learning_rate=LEARNING_RATE, dataset_train=train_data,
#          val_loader=val_loader, label_text_id_from_index_dict=region_ID_2_ylabel, true_val_labels=y_labels_val,
#          use_gpu=use_gpu)


max_epochs=200
epoch=1
criterion = nn.BCELoss()
while epoch <= max_epochs:
    print('\nEpoch: %d' % epoch)
    epoch = epoch +1
    valid_loss = 0
    y_true, y_pred = [], []

    train(epoch, train_data, BATCH_SIZE)

    true, pred, y_loss =  valid(val_loader, VALIDATION_BATCH_SIZE)

    # valid_sn, valid_sp = performance(y_true, y_pred)

    fpr, tpr, thresholds = metrics.roc_curve(true, pred)
    precision, recall, _ = metrics.precision_recall_curve(true, pred)
    auroc_score = metrics.auc(fpr, tpr)
    aupr = metrics.auc(recall, precision)
    print(auroc_score, " ", aupr)
