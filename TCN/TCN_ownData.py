# -*- coding: utf-8 -*-
"""
@Time    : 5/9/2022 2:30 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
# ======================
#     TCN Components
# ======================
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
# -------------------------------------------
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
DEVICE = "cpu"


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0,
                                           dilation=dilation))
        # here padding=0 because we are going to implement padding in next line
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
                                 self.pad, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs,
                                    1) if n_inputs != n_outputs else None  # this is the 1x1 Convolution network, used to make sure input size and output size are the same.
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, in_channel, out_channel, num_channels, kernel_size, dropouts):  # num_channels is the dialation at each residual block
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(in_channel, num_channels, kernel_size, dropouts)
        self.linear = nn.Linear(num_channels[-1], out_channel)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return o


def get_dataset(x, y):
    return TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())


def get_dataloader(x: np.array, y: np.array, batch_size: int, shuffle: bool = True, num_workers: int = 0):
    dataset = get_dataset(x, y)  # convert the numpy to torch standard TensorDataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)  # load and prepare the DataSet


def multivariate_data(dataset, target, start_index=0, end_index=None, history_size=20, target_size=-1, step=1, single_step=False):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]


def df_to_xyArray(inputDF, columnPos, max_colVal, min_colVla):  # convert the df to time series input array (X) and corrosponding output array (Y)
    columnIdx_name_ref = {}
    # only choose part of the dataframe
    features = inputDF[
        ['wind_speed', 'wind_angle', 'battery_voltage', 'battery_current', 'orientation_x', 'orientation_y',
         'orientation_z', 'orientation_w', 'velocity_x', 'velocity_y', 'velocity_z', 'angular_x', 'angular_y',
         'angular_z', 'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z', 'payload', 'altitude',
         'power']]
    featuresArray = features.values  # extract the values in DF, in terms of 2D array, same size with DF
    # normalise
    featuresArray[:, columnPos] = (featuresArray[:, columnPos] - min_colVla) / (max_colVal - min_colVla)  # normalise every flight's required column's data
    # ======================
    #     For reference on column idx vs column name for the featuresArray
    #   Column_index  |   column_name
    #       0         |      'wind_speed'
    #       1         |      'wind_angle'
    #       2         |      'battery_voltage'
    #       3         |      'battery_current'
    #       4         |      'orientation_x'
    #       5         |      'orientation_y'
    #       6         |      'orientation_z'
    #       7         |      'orientation_w'
    #       8         |      'velocity_x'
    #       9         |      'velocity_y'
    #       10        |      'velocity_z'
    #       11        |      'angular_x'
    #       12        |      'angular_y'
    #       13        |      'angular_z'
    #       14        |      'linear_acceleration_x'
    #       15        |      'linear_acceleration_y'
    #       16        |      'linear_acceleration_z'
    #       17        |      'payload'
    #       18        |      'altitude'
    #       19        |      'power'
    # ======================
    x_input = featuresArray[:, :-1].astype(float)  # set the input of the flight, excluding the last column of power
    supposed_y_output = featuresArray[:, -1].astype(float)
    for columnIdex in range(0, featuresArray.shape[1]):
        columnIdx_name_ref[columnIdex] = features.columns[columnIdex]
    x, y = multivariate_data(x_input, supposed_y_output, single_step=True)
    # Transpose the x, so that features is the row, time-sequence/step is column
    x = np.transpose(x, (0, 2, 1))  # 0th dimension stay at the 1st place, the 1st and 2nd dimension chance place.
    return x, y, columnIdx_name_ref

def load_xyDict_to_dataloaderDict(inputDict):
    dataLoaderDict = {}
    batchSize = 32
    for flightIdx_key, xyArray in inputDict.items():
        dataLoaderDict[flightIdx_key] = get_dataloader(xyArray[0], xyArray[1], batchSize)
    return dataLoaderDict


# ======================
#     Data cleaning
# ======================
# load data
df_raw = pd.read_csv(r"F:\githubClone\TCN\flights.csv", dtype={'altitude': str})  # because there is some row in "altitude" column that it varies, from 25 to 50 to 100 to 25
# remove data: position_x, position_y, position_z, speed, date, time_day, route
df_dropped = df_raw.drop(['position_x', 'position_y', 'position_z', 'speed', 'date', 'time_day', 'route'], axis=1)
# remove flights that has 0 altitude
df_dropped_altiZero = df_dropped.drop(df_dropped[df_dropped.altitude == str(0)].index)
# spilt dataframe into fixed altitude and variable altitude
df_mask = df_dropped_altiZero['altitude'] == '25-50-100-25'
df_dropped_altiZero_variAlti = df_dropped_altiZero[df_mask]
df_dropped_altiZero_fixAlti = df_dropped_altiZero[~df_mask]
# count total number of data points recorded for each flight (fixed altitude), make sure, every flight only has a single fixed altitude
flightDataPtCount = []
fixedAltiFlightNum = []
for flightidx in range(1, df_dropped_altiZero_fixAlti['flight'].max()+1):  # need to +1, because range(start, stop), will end at 'stop-1' in for loop.
    df_to_check = df_dropped_altiZero_fixAlti.loc[df_dropped_altiZero_fixAlti['flight'] == flightidx]
    if df_to_check.empty:
        pass
    else:
        assert df_to_check['altitude'].str.contains(df_to_check['altitude'].iloc[0]).all(), "There are different altitude for the same flight"  # ensure every flight only has a single fixed altitude
        flightDataPtCount.append(len(df_dropped_altiZero_fixAlti.loc[df_dropped_altiZero_fixAlti['flight'] == flightidx]))
        fixedAltiFlightNum.append(flightidx)  # this is used for time-series data for individual flights
#print(df_dropped_altiZero_fixAlti.head())
# ======================
#     Spilt flight data (fixed_altitude) into 60:20:20, training, validation, test
# ======================
np.random.seed(42)  # ensure spilt the same every time it runs during debugging
train_flightIdx, validate_flightIdx, test_flightIdx = np.split(np.array(fixedAltiFlightNum), [int(.6 * len(np.array(fixedAltiFlightNum))), int(.8 * len(np.array(fixedAltiFlightNum)))])

# combine the flight index of training and validation, so that normalisation can be done as a group.
test_validate_flight = np.hstack((train_flightIdx, validate_flightIdx))
# grab only "flight", "wind_speed", "wind_angle", "battery_voltage", "battery_current", "orientation_x", "orientation_y", "orientation_z", "orientation_w","velocity_x","velocity_y","velocity_z","angular_x","angular_y","angular_z", "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z","payload","altitude","power" for each flight number
test_validate_list = [df_dropped_altiZero_fixAlti.loc[df_dropped_altiZero_fixAlti['flight']==singleFlight, ['flight', 'wind_speed', 'wind_angle', 'battery_voltage', 'battery_current', 'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w', 'velocity_x', 'velocity_y', 'velocity_z', 'angular_x', 'angular_y', 'angular_z', 'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z', 'payload', 'altitude', 'power']] for singleFlight in test_validate_flight]
# all element in the list are DF, with same column names, just pick one to identify the index of the column that need to be normalised
ColumnToNormalise = column_index(test_validate_list[0], ['wind_speed', 'wind_angle', 'battery_voltage', 'battery_current'])

dataset_test_validate = np.concatenate([df_dropped_altiZero_fixAlti.loc[df_dropped_altiZero_fixAlti['flight']==singleFlight, ['wind_speed', 'wind_angle', 'battery_voltage', 'battery_current', 'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w', 'velocity_x', 'velocity_y', 'velocity_z', 'angular_x', 'angular_y', 'angular_z', 'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z', 'payload', 'altitude', 'power']] for singleFlight in test_validate_flight])
data_min = dataset_test_validate[:, ColumnToNormalise].min(axis=0)
data_max = dataset_test_validate[:, ColumnToNormalise].max(axis=0)

#build a dict to store the data for each flight for both training and validation DF, and normalize it.
train_data_dict = {}
valid_data_dict = {}
test_data_dict = {}
for eachFlight in test_validate_list:
    if eachFlight['flight'].iloc[0] in train_flightIdx:
        x, y, idx_name_ref = df_to_xyArray(eachFlight, ColumnToNormalise, data_max, data_min)  # x:(dim1,dim2,dim3) total of 1320 data points, 20 data points form a timeseries, 19 features in total. y: (dim1), input of 1320 set of 20 data points, leads to a power value.
        train_data_dict[eachFlight['flight'].iloc[0]] = (x, y)
    elif eachFlight['flight'].iloc[0] in validate_flightIdx:
        x, y, idx_name_ref = df_to_xyArray(eachFlight, ColumnToNormalise, data_max, data_min)
        valid_data_dict[eachFlight['flight'].iloc[0]] = (x, y)
    elif eachFlight['flight'].iloc[0] in test_flightIdx:
        x, y, idx_name_ref = df_to_xyArray(eachFlight, ColumnToNormalise, data_max, data_min)
        test_data_dict[eachFlight['flight'].iloc[0]] = (x, y)
# load the data dictionary to dataloader and store as dictionary for every flight
dataLoader_training_Dict = load_xyDict_to_dataloaderDict(train_data_dict)
dataLoader_validation_Dict = load_xyDict_to_dataloaderDict(valid_data_dict)
dataLoader_test_Dict = load_xyDict_to_dataloaderDict(test_data_dict)

# ======================
#     Configure and load TCN model
# ======================
overallInutChannel = 19  # total of 19 features
overallOutputChannel = 1
residualBlock_num = 6  # so a total of 6 blocks, this also controls the dilatation size for each residual block

# number of filters meaning number of kernel used is convolution layer in parallel?
num_filters = 64  # according to paper on "CVaR-based Flight Energy Risk Assessment for Multirotor UAVs using a Deep Energy Model"
outputChannelAfter_eachResidualBlock = [num_filters] * residualBlock_num
dropOut = 0.0  # according to model paper
torch.manual_seed(42)
model = TCN(overallInutChannel, overallOutputChannel, outputChannelAfter_eachResidualBlock, kernel_size=2, dropouts=dropOut)

# ======================
#     Training loop
# ======================
total_epochs = 50
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(ep):
    model.train()
    train_loss = 0
    np.random.seed(42)
    np.random.shuffle(train_flightIdx)  # now, the train_flightIdx is a numpy array that has been shuffled randomly
    for idx, flight in enumerate(train_flightIdx):  # train based on flight index
        # train(loop through) the portion of data in the data_loader that matches the flight index
        for batch_idx, (data_in, target) in enumerate(dataLoader_training_Dict[flight]):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            pred_out = model(data_in)  # data_in, is (N,C,L) format,N is batch size, C is number of features, L is the length of sequence (time-steps)
            # calculate loss
            loss = criterion(pred_out.squeeze(), target)
            # weight assignment
            loss.backward()
            # update model weights
            optimizer.step()
            train_loss += loss.item()
            # print(loss.item())
    return train_loss


def evaluate(X_data, name):
    model.eval()
    total_loss = 0.0
    count = 0
    Idx_list = None
    np.random.seed(42)
    if name == "Validation":
        Idx_list = validate_flightIdx
        np.random.shuffle(Idx_list)
    elif name == "Test":
        Idx_list = test_flightIdx
        np.random.shuffle(Idx_list)

    with torch.no_grad():
        loss_list = 0
        for idx, flight in enumerate(Idx_list):
            for batch_idx, (data_in, target) in enumerate(X_data[flight]):
                # compute the model output
                pred_out = model(data_in)  # data_in, is (N,C,L) format,N is batch size, C is number of features, L is the length of sequence (time-steps)
                # calculate loss
                loss = criterion(pred_out.squeeze(), target)
                loss_list += loss.item()
        return loss_list



for epIdx in range(1, total_epochs+1):
    train_loss = train(epIdx)
    print(epIdx, train_loss)
    vloss = evaluate(dataLoader_validation_Dict, name='Validation')
    # tloss = evaluate(dataLoader_test_Dict, name='Test')


# train_flightData = df_dropped_altiZero_fixAlti.loc[df_dropped_altiZero_fixAlti['flight'].isin(train_flightIdx)]
# validate_flightData = df_dropped_altiZero_fixAlti.loc[df_dropped_altiZero_fixAlti['flight'].isin(validate_flightIdx)]
# test_flightData = df_dropped_altiZero_fixAlti.loc[df_dropped_altiZero_fixAlti['flight'].isin(test_flightIdx)]
# print(train_flightData.head())



