import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, num_series, hidden, lag, kernel_size):
        super(DLinear, self).__init__()

        # Decompsition Kernel Size
        self.decompsition = series_decomp(kernel_size)

        self.Linear_Seasonal = nn.Conv1d(num_series, hidden, lag)
        self.Linear_Trend = nn.Conv1d(num_series, hidden, lag)
        self.Linear_Decoder = nn.Conv1d(hidden, 1, 1)
        

    def forward(self, x):
         X = X.transpose(2, 1)
        
        seasonal_init, trend_init = self.decompsition(X)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        X = seasonal_output + trend_output
        X = self.Linear_Decoder(X)

        return X.transpose(2, 1)

class cDLinear(nn.Module):
    def __init__(self, num_series, hidden, lag, kernel_size = 25):
        super(cDLinear, self).__init__()

        # Decompsition Kernel Size
        self.decompsition = series_decomp(kernel_size)

        self.p = num_series
        self.lag = lag

        # Set up networks.
        self.networks = nn.ModuleList([
            DLinear(num_series, hidden, lag, kernel_size)
            for _ in range(num_series)])
        

    def forward(self, x):

        return torch.cat([network(X) for network in self.networks], dim=2)
        
