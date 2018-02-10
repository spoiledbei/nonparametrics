#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:17:44 2018

@author: congshanzhang
kernel density estimation
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

################################################################################
############### Class for NonParametric Density Estimation #####################
################################################################################
class OneDimensionalDensity(object):
    
    def __init__(self, data ,bandwidth, kernel='gaussian', xlim = [0, 1]):
        """
        A constructor for initializing an object of the Density class.
        
        inputs: 1. data is one-dimensional np array
                2. bandwidth is {'plugIn','ruleOfThumb','CV'}
                3. kernel is {'uniform','gaussian','epanechnikov','biweight'}
                4. xlim is the x-axis limits
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.n = len(data)
        self.xMin = xlim[0]
        self.xMax = xlim[1]
        self.data = data
        
    def xlinspace(self,num):
        xPlot = np.linspace(self.xMin, self.xMax, num=num, endpoint=True)
        return xPlot
    
    def kernel(self,x,dat,h):
        """
        For each fixed point x, calculate kernel values for all the data;
        inputs: 
            1. a point x
            2. dat is a np.array;
            3. h is the bandwidth.
        """
        xx = (dat - x)/h
        if self.kernel == 'uniform':
            K = 1.0/2.0 * np.int64(1-np.abs(xx)>=0)
        
        if self.kernel == 'gaussian':      
            K = 1.0/(np.sqrt(2.0*np.pi)) * np.exp(-xx**2/2.0)
            
        if self.kernel == 'epanechnikov':
            K = 3.0/4.0*(1-xx**2) * np.int64(1-np.abs(xx)>=0)
            
        if self.kernel == 'biweight':
            K = 15.0/16.0 *(1-xx**2)**2 * np.int64(1-np.abs(xx)>=0)
        return K   
    
    
    def leaveOneOut(self,x,i,h): # leave-one-out estimator
        """
        kernel density estimator, leaving the i-th observation out
        """
        dat = np.delete(self.data,i) 
        return 1.0/((self.n-1)*h) * np.sum( self.kernel(x,dat,h) )
            
    
    def bandwidthChoice(self,num):
        if self.bandwidth == 'ruleOfThumb':
            h = 1.06*self.n**(-0.2)*np.sqrt(np.var(self.data))
        
        if self.bandwidth == 'CV':
            xPlot = self.xlinspace(num)
            dx = (self.xMax - self.xMin)/num
            
            def obj1(h): # first term in objective function
                s = 0
                for x in xPlot:
                    s = s + np.sum(np.outer(self.kernel(x,self.data,h),self.kernel(x,self.data,h))) * dx           
                return 1.0/(self.n**2 * h**2) * s 
            
            def obj2(h): # second term in objective function
                s = 0
                for i in range(self.n):      
                    s = s + self.leaveOneOut(self.data[i],i,h)
                return 1.0/self.n * s
            
            def obj(h): 
                return obj1(h) + obj2(h)
            
            x0 = 0.1 
            res = minimize(obj, x0, method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
            h = res.x
        return h     
        
    
    def fit(self,num):
        """
        Implement nonparametric kernel density estimation
        inputs: 
            1. num is the number of grid we split the x-axis
        """
        xPlot = self.xlinspace(num)
        f_hat = np.array([])
        for x in xPlot:
            f = 1.0/(self.n*self.bandwidthChoice(num)) * np.sum(self.kernel(x,self.data,self.bandwidthChoice(num)))
            np.append(f_hat, f) 
        
        return f_hat
    
    
    def graph(self,num,fname=None):   
        xPlot = self.xlinspace(num)
        plt.plot(xPlot, self.fit(num),'-',label="kernel = '{0}'".format(self.kernel))
        plt.legend(loc='upper right')
        plt.xlabel('c')
        plt.savefig('/Users/Congshanzhang/Desktop/'+str(fname)+'.pdf', format='pdf', bbox_inches = 'tight',dpi=1200)
        plt.show()



