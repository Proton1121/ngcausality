import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

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
        

    def forward(self, X):
        
        seasonal_init, trend_init = self.decompsition(X)
        seasonal_output = self.Linear_Seasonal(seasonal_init.transpose(2,1))
        trend_output = self.Linear_Trend(trend_init.transpose(2,1))

        X = seasonal_output + trend_output
        X = self.Linear_Decoder(X)

        return X.transpose(2, 1)

class cDLinear(nn.Module):
    def __init__(self, num_series, hidden, lag, kernel_size = 25):
        super(cDLinear, self).__init__()

        self.p = num_series
        self.lag = lag

        # Set up networks.
        self.networks = nn.ModuleList([
            DLinear(num_series, hidden, lag, kernel_size)
            for _ in range(num_series)])
        

    def forward(self, X):

        return torch.cat([network(X) for network in self.networks], dim=2)
        
    def GC(self, threshold=True, ignore_lag=True):
        '''
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.
          ignore_lag: if true, calculate norm of weights jointly for all lags.

        Returns:
          GC: (p x p) or (p x p x lag) matrix. In first case, entry (i, j)
            indicates whether variable j is Granger causal of variable i. In
            second case, entry (i, j, k) indicates whether it's Granger causal
            at lag k.
        '''
        if ignore_lag:
            GC = [torch.norm(torch.cat([net.Linear_Seasonal.weight, net.Linear_Trend.weight], dim=0), dim=(0, 2)) 
               for net in self.networks]
        else:
            GC = [torch.norm(torch.cat([net.Linear_Seasonal.weight, net.Linear_Trend.weight], dim=0), dim=0) 
               for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0).int()
        else:
            return GC


def prox_update(network, lam, lr, penalty):
    '''
    Perform in place proximal update on first layer weight matrix.

    Args:
      network: MLP network.
      lam: regularization parameter.
      lr: learning rate.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    W1 = network.Linear_Seasonal.weight
    W2 = network.Linear_Trend.weight
    W_combined = torch.cat([network.Linear_Seasonal.weight, network.Linear_Trend.weight], dim=0)
    hidden, p, lag = W1.shape
    if penalty == 'GL':
        norm = torch.norm(W_combined, dim=(0, 2), keepdim=True)
        W1.data = ((W1 / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        W2.data = ((W2 / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'GSGL':
        norm = torch.norm(W_combined, dim=0, keepdim=True)
        W1.data = ((W1 / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        W2.data = ((W2 / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        W_combined = torch.cat([network.Linear_Seasonal.weight, network.Linear_Trend.weight], dim=0)
        norm = torch.norm(W_combined, dim=(0, 2), keepdim=True)
        W1.data = ((W1 / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        W2.data = ((W2 / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        for i in range(lag):
            norm = torch.norm(W_combined[:, :, :(i + 1)], dim=(0, 2), keepdim=True)
            W1.data[:, :, :(i+1)] = (
                (W1.data[:, :, :(i+1)] / torch.clamp(norm, min=(lr * lam)))
                * torch.clamp(norm - (lr * lam), min=0.0))
            W2.data[:, :, :(i+1)] = (
                (W2.data[:, :, :(i+1)] / torch.clamp(norm, min=(lr * lam)))
                * torch.clamp(norm - (lr * lam), min=0.0))
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def regularize(network, lam, penalty):
    '''
    Calculate regularization term for first layer weight matrix.

    Args:
      network: MLP network.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    W1 = network.Linear_Seasonal.weight
    W2 = network.Linear_Trend.weight
    W_combined = torch.cat([network.Linear_Seasonal.weight, network.Linear_Trend.weight], dim=0)
    hidden, p, lag = W1.shape
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(W_combined, dim=(0, 2)))
    elif penalty == 'GSGL':
        return lam * (torch.sum(torch.norm(W_combined, dim=(0, 2)))
                      + torch.sum(torch.norm(W_combined, dim=0)))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        return lam * sum([torch.sum(torch.norm(W_combined[:, :, :(i+1)], dim=(0, 2)))
                          for i in range(lag)])
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def ridge_regularize(network, lam):
    '''Apply ridge penalty at all subsequent layers.'''
    return lam * torch.sum(network.Linear_Decoder.weight ** 2) 


def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params

def train_model_ista(cdlinear, X, lr, max_iter, lam=0, lam_ridge=0, penalty='H',
                     lookback=5, check_every=100, verbose=1):
    '''Train model with Adam.'''
    lag = cdlinear.lag
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    train_loss_list = []

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    # Calculate smooth error.
    loss = sum([loss_fn(cdlinear.networks[i](X[:, :-1]), X[:, lag:, i:i+1])
                for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in cdlinear.networks])
    smooth = loss + ridge

    for it in range(max_iter):
        # Take gradient step.
        smooth.backward()
        for param in cdlinear.parameters():
            param.data = param - lr * param.grad

        # Take prox step.
        if lam > 0:
            for net in cdlinear.networks:
                prox_update(net, lam, lr, penalty)

        cdlinear.zero_grad()

        # Calculate loss for next iteration.
        loss = sum([loss_fn(cdlinear.networks[i](X[:, :-1]), X[:, lag:, i:i+1])
                    for i in range(p)])
        ridge = sum([ridge_regularize(net, lam_ridge) for net in cdlinear.networks])
        smooth = loss + ridge
        
        # Check progress.
        if (it + 1) % check_every == 0:
            # Add nonsmooth penalty.
            nonsmooth = sum([regularize(net, lam, penalty)
                             for net in cdlinear.networks])
            mean_loss = (smooth + nonsmooth) / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)
                print('nonsmooth = %f' % (nonsmooth / p))
                print('smooth = %f' % (smooth / p))
                print('Variable usage = %.2f%%'
                      % (100 * torch.mean(cdlinear.GC().float())))

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(cdlinear)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print('Stopping early')
                break

    # Restore best model.
    restore_parameters(cdlinear, best_model)

    return train_loss_list
