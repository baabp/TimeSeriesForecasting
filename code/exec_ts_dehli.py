import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.utils.tensorboard import SummaryWriter
from torchsummaryX import summary

from code.dataset.ts_dataset import TimeSeriesDatasetDelhi
from code.models.utils import EarlyStopping
from code.models.ts_lstm import LSTM


def train_one_epoch(model, criterion, optimizer, X, y, batch_size, device, epoch=None, print_freq=None, writer=None):
    n_batches_train = X.shape[0] // batch_size + 1
    train_loss = 0.
    model.train()
    for batch in range(n_batches_train):
        start = batch * batch_size
        end = start + batch_size
        X_batch = torch.Tensor(X[start:end]).to(device)
        y_batch = torch.Tensor(y[start:end]).to(device)
        pred_batch = model(X_batch)
        loss = criterion(pred_batch.view(-1), y_batch.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= n_batches_train
    writer.add_scalar('loss/train', train_loss, epoch * X.shape[0])
    return train_loss


def val_one_epoch(model, criterion, X, y, batch_size, device, epoch, print_freq=None, writer=None):
    n_batches_val = X.shape[0] // batch_size + 1
    val_loss = 0.
    model.eval()
    for batch in range(n_batches_val):
        start = batch * batch_size
        end = start + batch_size
        X_batch = torch.Tensor(X[start:end]).to(device)
        y_batch = torch.Tensor(y[start:end]).to(device)
        pred_batch = model(X_batch)
        loss = criterion(pred_batch.view(-1), y_batch.view(-1))

        val_loss += loss.item()
    val_loss /= n_batches_val
    writer.add_scalar('loss/val', val_loss, X.shape[0] * epoch)

    return val_loss


def main():
    # Dataset

    # args
    path_data_train = 'data/DailyDelhiClimate/DailyDelhiClimateTrain.csv'
    path_data_test = 'data/DailyDelhiClimate/DailyDelhiClimateTest.csv'
    look_back = 6
    test_size = 0.2

    # args for output data
    dir_log = 'data/DailyDelhiClimate/logs'

    df_train = pd.read_csv(path_data_train)
    df_test = pd.read_csv(path_data_test)

    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])

    time_series_dataset_delhi = TimeSeriesDatasetDelhi(df_train=df_train,
                                                       df_test=df_test)
    time_series_dataset_delhi.remove_null_row()
    time_series_dataset_delhi.transform_cols()
    time_series_dataset_delhi.log_cols()
    X_train, y_train, list_datetime_train, X_test, y_test, list_datetime_test = \
        time_series_dataset_delhi.get_ts_dataset(look_back=look_back)

    X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train, test_size=test_size, shuffle=False)

    # set parameters for torch
    np.random.seed(123)
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 2000
    batch_size = 64

    n_batches_val = X_val.shape[0] // batch_size + 1
    learning_rate = 0.01

    input_size = 4
    hidden_size = 10
    num_layers = 1
    num_classes = 4

    hist = {'loss': [], 'val_loss': []}
    es = EarlyStopping(patience=10, verbose=1)

    model = LSTM(num_classes, input_size, hidden_size, num_layers).to(device)

    criterion = nn.MSELoss()
    optimizer = optimizers.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = optimizers.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # print out model structures
    print(model)
    inputs = torch.zeros((batch_size, look_back, input_size)).to(device)  # [length, batch_size]
    # print(summary(model, inputs))

    # tensorboard
    # tensorboard
    writer = SummaryWriter(log_dir=dir_log)
    writer.add_graph(model, torch.Tensor(X_train[:batch_size]).to(device))

    # train
    for epoch in range(num_epochs):
        writer.add_scalar('train/learning_rate', lr_scheduler.get_lr()[0], epoch)
        X_train_sf, y_train_sf = shuffle(X_train, y_train)
        train_loss = train_one_epoch(model=model,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     X=X_train_sf,
                                     y=y_train_sf,
                                     epoch=epoch,
                                     batch_size=batch_size,
                                     device=device,
                                     writer=writer)
        lr_scheduler.step()

        val_loss = val_one_epoch(model=model,
                                 criterion=criterion,
                                 X=X_val,
                                 y=y_val,
                                 epoch=epoch,
                                 batch_size=batch_size,
                                 device=device,
                                 writer=writer)
        hist['loss'].append(train_loss)
        hist['val_loss'].append(val_loss)

        print('epoch: {}, loss: {:.3}, val_loss: {:.3f}'.format(
            epoch + 1,
            train_loss,
            val_loss
        ))

        if es(val_loss):
            break


if __name__ == '__main__':
    main()
