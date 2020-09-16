import torch.nn as nn


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
