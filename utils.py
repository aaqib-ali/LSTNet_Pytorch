import numpy as np
import torch


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_Utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize=2):
        self.cuda = cuda
        self.P = window
        self.h = horizon
        self.raw_data = np.loadtxt(file_name, delimiter=',')
        self.dat = np.zeros(self.raw_data.shape)
        self.n, self.m = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n))

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        if self.cuda:
            self.scale = self.scale.cuda()

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if normalize == 0:
            self.dat = self.raw_data

        if normalize == 1:
            self.dat = self.raw_data / np.max(self.raw_data)

        # normalized by the maximum value of each row(sensor).
        if normalize == 2:
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.raw_data[:, i]))
                self.dat[:, i] = self.raw_data[:, i] / np.max(np.abs(self.raw_data[:, i]))

    def _split(self, train, valid):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set)
        self.valid = self._batchify(valid_set)
        self.test = self._batchify(test_set)

    def _batchify(self, idx_set):

        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            if self.cuda:
                X = X.cuda()
                Y = Y.cuda()
            yield X, Y
            start_idx += batch_size
