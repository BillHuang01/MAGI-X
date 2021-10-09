import torch
import numpy as np
import matplotlib.pyplot as plt

from . import gpm
from . import integrator

torch.set_default_dtype(torch.double)

class odeinfer(object):
    def __init__(self, y, t, dynamic):
        '''
        Constructor of the odeINFER module
        Args:
            y: 2d observations, missing data is nan
            t: 1d observation time
        '''
        if (not torch.is_tensor(y)):
            y = torch.tensor(y).double()
        if (not torch.is_tensor(t)):
            t = torch.tensor(t).double().squeeze()
        self.n, self.p = y.size()
        self.y = y
        # standardize time
        # self.t = t
        self.t_range = [torch.min(t).item(), torch.max(t).item()]
        self.t = (t-self.t_range[0]) / (self.t_range[1]-self.t_range[0]) / 20 * (self.n-1)
        self.fOde = dynamic
        # turn off dynamic gradient unless in training
        for param in self.fOde.parameters():
            param.requires_grad_(False)
        self._gpPreprocess()

    def _gpPreprocess(self):
        '''
        GP preprocessing of the Input Data
        '''
        self.gpModels = []
        for i in range(self.p):
            # available observation index
            aIdx = ~torch.isnan(self.y[:,i]) 
            yi = self.y[aIdx,i]
            ti = self.t[aIdx]
            mean, outputscale, noisescale, kernel = gpm.GPTrain(ti, yi)
            self.gpModels.append({
                'aIdx':aIdx, # non-missing data index
                'mean':mean,
                'kernel':kernel,
                'outputscale':outputscale,
                'noisescale':noisescale
            })

    def ls(self, nEpoch = 5000, verbose = False, visualization = False, returnX = False):
        '''
        Least Squares Inference of ODE Parameters
        '''

        # obtain features from gpModels
        x = torch.empty(self.n, self.p).double()
        dxdtGP = torch.empty(self.n, self.p).double()
        for i in range(self.p):
            aIdx = self.gpModels[i]['aIdx']
            yi = self.y[aIdx,i]
            ti = self.t[aIdx]
            kernel = self.gpModels[i]['kernel']
            mean = self.gpModels[i]['mean']
            outputscale = self.gpModels[i]['outputscale']
            noisescale = self.gpModels[i]['noisescale']
            alpha = torch.inverse(kernel.C(ti)+noisescale/outputscale*torch.eye(ti.size(0))) @ (yi-mean)
            x[:,i] = mean + kernel.C(self.t,ti) @ alpha
            dxdtGP[:,i] = kernel.dC_dx1(self.t,ti) @ alpha

        # optimize over theta
        lossVal = np.zeros(nEpoch)
        for param in self.fOde.parameters():
                param.requires_grad_(True)
        theta_optimizer = torch.optim.Adam(self.fOde.parameters(), lr = 1e-3)
        for epoch in range(nEpoch):
            theta_optimizer.zero_grad()
            mse = torch.zeros(self.p)
            dxdtOde = self.fOde(x)
            for i in range(self.p):
                dxdtError = dxdtOde[:,i] - dxdtGP[:,i]
                mse[i] = torch.mean(torch.square(dxdtError))
            theta_loss = torch.sum(mse)
            lossVal[epoch] = theta_loss.item()
            if (verbose and ((epoch+1) % 500 == 0)):
                print('%d/%d iteration: %.6f' %(epoch+1,nEpoch,lossVal[epoch]))
            theta_loss.backward()
            theta_optimizer.step()

        # turning gradient information off
        for param in self.fOde.parameters():
            param.requires_grad_(False)

        if (visualization):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(np.arange(nEpoch),lossVal,lw=2)
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Training MSE')
            plt.show()

        if (returnX):
            return (x.numpy())


    def map(self, nEpoch = 2500, verbose = False, returnX = False):

        '''
        MAP Inference of ODE Parameters
        '''

        # obtain features from gpModels
        gpCov = []
        u = torch.empty(self.n, self.p).double()
        x = torch.empty(self.n, self.p).double()
        dxdtGP = torch.empty(self.n, self.p).double()
        for i in range(self.p):
            # get observation data
            aIdx = self.gpModels[i]['aIdx']
            yi = self.y[aIdx,i]
            ti = self.t[aIdx]
            # get hyperparameters
            kernel = self.gpModels[i]['kernel']
            mean = self.gpModels[i]['mean']
            outputscale = self.gpModels[i]['outputscale']
            noisescale = self.gpModels[i]['noisescale']
            # Compute GP prior covariance matrix
            Cxx = kernel.C(self.t) + 1e-6 * torch.eye(self.n)
            LC = torch.cholesky(Cxx)
            LCinv = torch.inverse(LC)
            Cinv = LCinv.T @ LCinv
            # obtain initial values
            Cxz = kernel.C(self.t, ti)
            S = Cxz @ torch.inverse(kernel.C(ti)+noisescale/outputscale*torch.eye(ti.size(0)))
            xmean = mean + S @ (yi - mean)
            # obtain derivative information
            dCdx1 = kernel.dC_dx1(self.t)
            dCdx2 = kernel.dC_dx2(self.t)
            d2Cdx1dx2 = kernel.d2C_dx1dx2(self.t)
            m = dCdx1 @ Cinv
            K = d2Cdx1dx2 - dCdx1 @ Cinv @ dCdx2 + 1e-6 * torch.eye(self.n)
            Kinv = torch.inverse(K)
            gpCov.append({'LC':LC,'m':m,'Kinv':Kinv})
            x[:,i] = xmean
            u[:,i] = LCinv @ (xmean - mean) / np.sqrt(outputscale)
            dxdtGP[:,i] = m @ (xmean - mean)
        
        # optimize the initial theta
        for param in self.fOde.parameters():
            param.requires_grad_(True)
        theta_optimizer = torch.optim.Adam(self.fOde.parameters(), lr = 5e-3)
        prev_theta_loss = np.Inf
        for tt in range(500): # max 500 iteration for initialization
            theta_optimizer.zero_grad()
            lkh = torch.zeros(self.p)
            dxdtOde = self.fOde(x)
            for i in range(self.p):
                outputscale = self.gpModels[i]['outputscale']
                dxdtError = dxdtOde[:,i] - dxdtGP[:,i]
                lkh[i] = -0.5/outputscale * dxdtError.T @ gpCov[i]['Kinv'] @ dxdtError 
            theta_loss = -torch.sum(lkh)
            theta_loss.backward()
            theta_optimizer.step()
            if (np.nan_to_num((prev_theta_loss-theta_loss.item())/abs(prev_theta_loss),nan=1) > 1e-3):
                prev_theta_loss = theta_loss.item()
            else:
                break
        # detach theta gradient
        for param in self.fOde.parameters():
            param.requires_grad_(False)

        # optimize over all parameters
        lossVal = np.zeros(nEpoch)
        for epoch in range(nEpoch):
            # optimize theta
            theta_lr = 5e-3 * np.power(epoch+1,-0.6)
            for param in self.fOde.parameters():
                param.requires_grad_(True)
            theta_optimizer = torch.optim.Adam(self.fOde.parameters(), lr = theta_lr)
            for tt in range(1):
                theta_optimizer.zero_grad()
                lkh = torch.zeros(self.p)
                dxdtOde = self.fOde(x)
                for i in range(self.p):
                    outputscale = self.gpModels[i]['outputscale']
                    dxdtError = dxdtOde[:,i] - dxdtGP[:,i]
                    lkh[i] = -0.5/outputscale * dxdtError.T @ gpCov[i]['Kinv'] @ dxdtError
                theta_loss = -torch.sum(lkh)
                # print(theta_loss.item())
                theta_loss.backward()
                theta_optimizer.step()
            # detach gradient
            for param in self.fOde.parameters():
                param.requires_grad_(False)

            # optimize u (x after Cholesky transformation)
            # u_lr = 5e-4 # * np.power(epoch+1,-0.6)
            u_lr = 5e-2 * np.power(epoch+501,-0.6)
            u.requires_grad_(True)
            u_optimizer = torch.optim.Adam([u], lr = u_lr)
            for ut in range(1):
                u_optimizer.zero_grad()
                # reconstruct x
                x = torch.empty(self.n,self.p).double()
                dxdtGP = torch.empty(self.n,self.p).double()
                for i in range(self.p):
                    mean = self.gpModels[i]['mean']
                    outputscale = self.gpModels[i]['outputscale']
                    x[:,i] = mean + np.sqrt(outputscale) * gpCov[i]['LC'] @ u[:,i]
                    dxdtGP[:,i] = gpCov[i]['m'] @ (x[:,i] - mean)
                dxdtOde = self.fOde(x)
                lkh = torch.zeros((self.p, 3))
                for i in range(self.p):
                    # p(X[I] = x[I]) = P(U[I] = u[I])
                    lkh[i,0] = -0.5 * torch.mean(torch.square(u[:,i]))
                    # p(Y[I] = y[I] | X[I] = x[I])
                    aIdx = self.gpModels[i]['aIdx']
                    noisescale = self.gpModels[i]['noisescale']
                    lkh[i,1] = -0.5/noisescale * torch.mean(torch.square(x[aIdx,i]-self.y[aIdx,i])) 
                    # p(X'[I]=f(x[I],theta)|X(I)=x(I))
                    outputscale = self.gpModels[i]['outputscale']
                    dxdtError = dxdtOde[:,i] - dxdtGP[:,i]
                    lkh[i,2] = -0.5/outputscale * dxdtError.T @ gpCov[i]['Kinv'] @ dxdtError / self.n
                u_loss = -torch.sum(lkh)
                # print(ut, u_loss.item())
                u_loss.backward()
                u_optimizer.step()
            u.requires_grad_(False) # detach gradient information

            # store loss value
            lossVal[epoch] = u_loss.item()
            if (verbose and ((epoch+1) % 100 == 0)):
                print('%d/%d iteration: %.6f' %(epoch+1,nEpoch,lossVal[epoch]))

            # compute x and dxdtGP
            x = torch.empty(self.n,self.p).double()
            dxdtGP = torch.empty(self.n,self.p).double()
            for i in range(self.p):
                mean = self.gpModels[i]['mean']
                outputscale = self.gpModels[i]['outputscale']
                x[:,i] = mean + np.sqrt(outputscale) * gpCov[i]['LC'] @ u[:,i]
                dxdtGP[:,i] = gpCov[i]['m'] @ (x[:,i] - mean)
            # optimize noisescale parameter
            for i in range(self.p):
                aIdx = self.gpModels[i]['aIdx']
                self.gpModels[i]['noisescale'] = torch.mean(torch.square(x[aIdx,i]-self.y[aIdx,i])).item()

        if (returnX):
            return (x.numpy())

    def sgld(self, nEpoch = 2500, verbose = False, returnX = False):

        '''
        Stochastic Gradient via Langevin Dynamics
        '''

        # obtain features from gpModels
        gpCov = []
        u = torch.empty(self.n, self.p).double()
        x = torch.empty(self.n, self.p).double()
        dxdtGP = torch.empty(self.n, self.p).double()
        for i in range(self.p):
            # get observation data
            aIdx = self.gpModels[i]['aIdx']
            yi = self.y[aIdx,i]
            ti = self.t[aIdx]
            # get hyperparameters
            kernel = self.gpModels[i]['kernel']
            mean = self.gpModels[i]['mean']
            outputscale = self.gpModels[i]['outputscale']
            noisescale = self.gpModels[i]['noisescale']
            # Compute GP prior covariance matrix
            Cxx = kernel.C(self.t) + 1e-6 * torch.eye(self.n)
            LC = torch.cholesky(Cxx)
            LCinv = torch.inverse(LC)
            Cinv = LCinv.T @ LCinv
            # obtain initial values
            Cxz = kernel.C(self.t, ti)
            S = Cxz @ torch.inverse(kernel.C(ti)+noisescale/outputscale*torch.eye(ti.size(0)))
            xmean = mean + S @ (yi - mean)
            xvar = Cxx - S @ Cxz.T
            LU = LCinv @ torch.cholesky(xvar)
            # obtain derivative information
            dCdx1 = kernel.dC_dx1(self.t)
            dCdx2 = kernel.dC_dx2(self.t)
            d2Cdx1dx2 = kernel.d2C_dx1dx2(self.t)
            m = dCdx1 @ Cinv
            K = d2Cdx1dx2 - dCdx1 @ Cinv @ dCdx2 + 1e-6 * torch.eye(self.n)
            Kinv = torch.inverse(K)
            gpCov.append({'LC':LC,'LU':LU,'m':m,'Kinv':Kinv})
            x[:,i] = xmean
            u[:,i] = LCinv @ (xmean - mean) / np.sqrt(outputscale)
            dxdtGP[:,i] = m @ (xmean - mean)
        
        # optimize the initial theta
        for param in self.fOde.parameters():
            param.requires_grad_(True)
        theta_optimizer = torch.optim.Adam(self.fOde.parameters(), lr = 5e-3)
        prev_theta_loss = np.Inf
        for tt in range(500): # max 500 iteration for initialization
            theta_optimizer.zero_grad()
            lkh = torch.zeros(self.p)
            dxdtOde = self.fOde(x)
            for i in range(self.p):
                outputscale = self.gpModels[i]['outputscale']
                dxdtError = dxdtOde[:,i] - dxdtGP[:,i]
                lkh[i] = -0.5/outputscale * dxdtError.T @ gpCov[i]['Kinv'] @ dxdtError 
            theta_loss = -torch.sum(lkh)
            theta_loss.backward()
            theta_optimizer.step()
            if (np.nan_to_num((prev_theta_loss-theta_loss.item())/abs(prev_theta_loss),nan=1) > 1e-3):
                prev_theta_loss = theta_loss.item()
            else:
                break
        # detach theta gradient
        for param in self.fOde.parameters():
            param.requires_grad_(False)

        # optimize over all parameters
        lossVal = np.zeros(nEpoch)
        xmean = torch.zeros(self.n, self.p).double()
        u_lr_sum = 0
        for epoch in range(nEpoch):
            # optimize theta
            theta_lr = 5e-3 * np.power(epoch+1,-0.6)
            for param in self.fOde.parameters():
                param.requires_grad_(True)
            theta_optimizer = torch.optim.Adam(self.fOde.parameters(), lr = theta_lr)
            for tt in range(1):
                theta_optimizer.zero_grad()
                lkh = torch.zeros(self.p)
                dxdtOde = self.fOde(x)
                for i in range(self.p):
                    outputscale = self.gpModels[i]['outputscale']
                    dxdtError = dxdtOde[:,i] - dxdtGP[:,i]
                    lkh[i] = -0.5/outputscale * dxdtError.T @ gpCov[i]['Kinv'] @ dxdtError
                theta_loss = -torch.sum(lkh)
                # print(theta_loss.item())
                theta_loss.backward()
                theta_optimizer.step()
            # detach gradient
            for param in self.fOde.parameters():
                param.requires_grad_(False)

            # optimize u (x after Cholesky transformation)
            u_lr = 5e-2 * np.power(epoch+501,-0.6)
            u.requires_grad_(True)
            u_optimizer = torch.optim.Adam([u], lr = u_lr)
            for ut in range(1):
                u_optimizer.zero_grad()
                # reconstruct x
                x = torch.empty(self.n,self.p).double()
                dxdtGP = torch.empty(self.n,self.p).double()
                for i in range(self.p):
                    mean = self.gpModels[i]['mean']
                    outputscale = self.gpModels[i]['outputscale']
                    x[:,i] = mean + np.sqrt(outputscale) * gpCov[i]['LC'] @ u[:,i]
                    dxdtGP[:,i] = gpCov[i]['m'] @ (x[:,i] - mean)
                dxdtOde = self.fOde(x)
                lkh = torch.zeros((self.p,3))
                for i in range(self.p):
                    # p(X[I] = x[I]) = P(U[I] = u[I])
                    lkh[i,0] = -0.5 * torch.mean(torch.square(u[:,i]))
                    # p(Y[I] = y[I] | X[I] = x[I])
                    aIdx = self.gpModels[i]['aIdx']
                    noisescale = self.gpModels[i]['noisescale']
                    lkh[i,1] = -0.5/noisescale * torch.mean(torch.square(x[aIdx,i]-self.y[aIdx,i])) 
                    # p(X'[I]=f(x[I],theta)|X(I)=x(I))
                    outputscale = self.gpModels[i]['outputscale']
                    dxdtError = dxdtOde[:,i] - dxdtGP[:,i]
                    lkh[i,2] = -0.5/outputscale * dxdtError.T @ gpCov[i]['Kinv'] @ dxdtError / self.n
                u_loss = -torch.sum(lkh)
                # print(ut, u_loss.item())
                u_loss.backward()
                u_optimizer.step()
            u.requires_grad_(False) # detach gradient information
            for i in range(self.p):
                u[:,i] = u[:,i] + u_lr * gpCov[i]['LU'] @ torch.randn(self.n).double()

            # store loss value
            lossVal[epoch] = u_loss.item()
            if (verbose and ((epoch+1) % 100 == 0)):
                print('%d/%d iteration: %.6f' %(epoch+1,nEpoch,lossVal[epoch]))

            # compute x and dxdtGP
            u_lr_sum = u_lr_sum + u_lr
            x = torch.empty(self.n,self.p).double()
            dxdtGP = torch.empty(self.n,self.p).double()
            for i in range(self.p):
                mean = self.gpModels[i]['mean']
                outputscale = self.gpModels[i]['outputscale']
                x[:,i] = mean + np.sqrt(outputscale) * gpCov[i]['LC'] @ u[:,i]
                dxdtGP[:,i] = gpCov[i]['m'] @ (x[:,i] - mean)
                xmean[:,i] = (1 - u_lr/u_lr_sum) * xmean[:,i] + (u_lr/u_lr_sum) * x[:,i]
            # optimize noise parameter
            for i in range(self.p):
                aIdx = self.gpModels[i]['aIdx']
                self.gpModels[i]['noisescale'] = torch.mean(torch.square(x[aIdx,i]-self.y[aIdx,i])).item()

        if (returnX):
            return (xmean.numpy())

    def predict(self, x0, ts, **params):
        # standardize time
        ts = (ts-self.t_range[0]) / (self.t_range[1]-self.t_range[0]) / 20 * (self.n-1)
        itg = integrator.RungeKutta(self.fOde)
        Xs = itg.forward(x0, ts, **params)
        return (Xs.numpy())     
