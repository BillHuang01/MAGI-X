import torch
import numpy as np
import scipy.special as fun

torch.set_default_dtype(torch.double)

class Bessel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, nu):
        ctx._nu = nu
        ctx.save_for_backward(inp)
        return (torch.from_numpy(np.array(fun.kv(nu,inp.detach().numpy()))))

    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        nu = ctx._nu
        grad_in = grad_out.numpy() * np.array(fun.kvp(nu,inp.detach().numpy()))
        return (torch.from_numpy(grad_in), None)

class Matern(object):

    # has_lengthscale = True

    def __init__(self, nu = 2.01, lengthscale = 1e-1, **kwargs):
        # super(Matern,self).__init__(**kwargs)
        self.nu = nu
        self.log_lengthscale = torch.tensor(np.log(lengthscale))
        self.log_lengthscale.requires_grad_(True)

    def _set_lengthscale(self, lengthscale):
        self.log_lengthscale = torch.tensor(np.log(lengthscale))

    def lengthscale(self):
        return (torch.exp(self.log_lengthscale).item())

    def forward(self, x1, x2 = None, **params):
        lengthscale = torch.exp(self.log_lengthscale)
        x1 = x1.squeeze()
        if x2 is None: x2 = x1
        r_ = (x1.reshape(-1,1) - x2.reshape(1,-1)).abs()
        r_ = np.sqrt(2.*self.nu) * r_ / lengthscale
        # handle limit at 0, allows more efficient backprop
        r_ = r_.clamp_(1e-15) 
        C_ = np.power(2,1-self.nu)*np.exp(-fun.loggamma(self.nu))*torch.pow(r_,self.nu)
        C_ = C_ * Bessel.apply(r_,self.nu)
        return (C_)

    def C(self, x1, x2 = None):
        return (self.forward(x1,x2).detach())

    def dC_dx1(self, x1, x2 = None):
        lengthscale = torch.exp(self.log_lengthscale)
        x1 = x1.squeeze()
        if x2 is None: x2 = x1
        with torch.no_grad():
            C_ = self.C(x1, x2)
            dist_ = (x1.reshape(-1,1) - x2.reshape(1,-1))
            r_ = dist_.abs()
            r_ = np.sqrt(2.*self.nu) * r_ / lengthscale
            r_ = r_.clamp_(1e-15)
            dC_ = C_ * (self.nu/r_ + fun.kvp(self.nu,r_)/fun.kv(self.nu,r_))
            dC_ = dC_ * np.sqrt(2*self.nu) / lengthscale
            # limit at 0 is taken care by the sign function
            dC_ = dC_ * torch.sign(dist_)
        return (dC_)

    def dC_dx2(self, x1, x2 = None):
        return (-self.dC_dx1(x1,x2))

    def d2C_dx1dx2(self, x1, x2 = None):
        lengthscale = torch.exp(self.log_lengthscale)
        x1 = x1.squeeze()
        if (x2 is None): x2 = x1
        with torch.no_grad():
            C_ = self.C(x1, x2)
            dist_ = (x1.reshape(-1,1) - x2.reshape(1,-1))
            r_ = dist_.abs()
            r_ = np.sqrt(2.*self.nu) * r_ / lengthscale
            r_ = r_.clamp_(1e-15)
            cons_part1 = self.nu * (self.nu - 1) / torch.square(r_)
            cons_part2 = 2. * self.nu/r_ * fun.kvp(self.nu,r_)/fun.kv(self.nu,r_)
            cons_part3 = fun.kvp(self.nu,r_,n=2)/fun.kv(self.nu,r_)
            dC2_ = C_ * (cons_part1 + cons_part2 + cons_part3)
            dC2_at_zero = -0.5 / (self.nu - 1) # handle 1st and 2nd derivative limit at 0
            dC2_[dist_ == 0] = dC2_at_zero
            dC2_ = dC2_ * 2. * self.nu / torch.square(lengthscale).double()
            dC2_ = -dC2_
        return (dC2_)


def GPTrain(train_x, train_y, kernel = None, noisy = True, max_iter = 500, verbose = False, eps=1e-6):
    # preprocess input data
    n = train_x.size(0)
    # normalized x to 0 and 1
    x_range = [torch.min(train_x).item(), torch.max(train_x).item()]
    train_x = (train_x - x_range[0]) / (x_range[1] - x_range[0])
    # set up kernel
    if (kernel is None):
        kernel = Matern(nu = 2.01, lengthscale = 1./(n-1))
    # set up optimizer
    if (noisy):
        # lambda = noise/outputscale
        log_lambda = torch.tensor(np.log(1e-1))
        log_lambda.requires_grad_(True)
        optimizer = torch.optim.Adam([kernel.log_lengthscale,log_lambda], lr=1e-2)
    else:
        # nugget term to avoid numerical issue
        log_lambda = torch.tensor(np.log(1e-6))
        optimizer = torch.optim.Adam([kernel.log_lengthscale], lr=1e-2)
    # training
    prev_loss = np.Inf
    for i in range(max_iter):
        R = kernel.forward(train_x) + torch.exp(log_lambda) * torch.eye(n)
        e,v = torch.eig(R, eigenvectors = True)
        e = e[:,0] # eigenvalues
        a = v.T @ torch.ones(n)
        b = v.T @ train_y
        mean = ((a/e).T @ b) / ((a/e).T @ a)
        d = v.T @ (train_y - mean)
        outputscale = 1./n * (d/e).T @ d
        loss = torch.log(outputscale) + torch.mean(torch.log(e))
        loss = loss + 1e6 * (log_lambda < np.log(1e-6)) # penealty for log lambda
        loss = loss + 1e6 * (kernel.log_lengthscale > 0) # penalty for lengthscale
        loss.backward()
        optimizer.step()
        # early termination check every 10 iterations
        if ((i+1)%10 == 0):
            if (verbose):
                print('Iter %d/%d - Loss: %.3f' % (i+1, max_iter, loss.item()))
            if (np.nan_to_num((prev_loss-loss.item())/abs(prev_loss),nan=1) > eps):
                prev_loss = loss.item()
            else:
                if (verbose): print('Early Termination!')
                break
    R = kernel.forward(train_x) + torch.exp(log_lambda) * torch.eye(n)
    Rinv = torch.inverse(R)
    ones = torch.ones(n)
    mean = ((ones.T @ Rinv @ train_y) / (ones.T @ Rinv @ ones)).item()
    outputscale = (1/n * (train_y - mean).T @ Rinv @ (train_y - mean)).item()
    noisescale = outputscale * torch.exp(log_lambda).item()
    # reset kernel lengthscale
    kernel._set_lengthscale(kernel.lengthscale()*(x_range[1] - x_range[0]))
    return (mean, outputscale, noisescale, kernel)