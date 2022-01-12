# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:32:55 2021

@author: nbaldelli
"""

import numpy as np
from numba import njit,jit
from numpy import linalg as alg
from matplotlib import pyplot as plt
import time

@njit #Crank-Nicholson evolution
def CRNI(psi,ham,dt):
    ide=np.eye(ham.shape[0])
    psin=alg.solve(ide+1j*ham*dt/2,(ide-1j*ham*dt/2)@psi)
    return psin

@njit #Lanczos evolution
def LANC(usq,ham,nmax,dt):
    usqN=np.zeros(usq.shape,dtype=np.complex128)
    for jj in range(len(usq[0,:])):
        psi=usq[:,jj]
        N=np.sqrt(np.vdot(psi,psi))
        q=np.zeros((len(psi),nmax+1),dtype=np.complex128); q[:,0]=psi/N
        beta=np.zeros((nmax+1,nmax+1),dtype=np.complex128)
        for k in range(nmax+1):  
            temp=ham@q[:,k]
            for j in range(k+1):
                beta[j,k]=np.vdot(q[:,j],temp)
                temp=temp-q[:,j]*beta[j,k]
            Nt=np.sqrt(np.vdot(temp,temp))
            if k!=nmax:
                beta[k+1,k]=Nt
                q[:,k+1]=temp/Nt
        lam,S=alg.eigh(beta)
        c=S@np.diag(np.exp(-1j*lam*dt))@S.T.conj()[:,0]*N
        usqN[:,jj]=q@c
    return usqN

@njit 
def PULSE(t,I,omega,env="N",ncyc=None): # A field pulse
    carr=I*np.sin(omega*t)
    if env=="S": #no envelope
        return carr*np.sin(omega*t/(2*ncyc))**2
    if env=="N": #sine squared envelope
        return carr 

@njit
def EPULSE(t,I,omega,env="N",ncyc=None): #E field pulse 
    if env=="N": #no envelope
        return -I*omega*np.cos(omega*t)
    if env=="S": #sine squared envelope
        return -I*omega*np.cos(omega*t)*np.sin(omega*t/(2*ncyc))**2-I*(omega/(2*ncyc))*np.sin(omega*t)*2*np.sin(omega*t/(2*ncyc))*np.cos(omega*t/(2*ncyc))

# @njit
def NWIRE(N,j,alpha,a,B,mu,delta,A,BC="O",loc=False,dis=np.array([[False,0],[0,0]]),lp=0): #nanowire hamiltonian
    bst=0.001#time reversal symmetry breaking term
    kine=np.diag(np.repeat(-mu-2*j,N))+np.diag(j*np.exp(1j*A+1j*bst),k=1)+np.diag(j*np.exp(-1j*A-1j*bst),k=-1)        
    # kine=np.diag(np.repeat(-mu-2*j,N))+np.diag(np.repeat(j*np.exp(1j*A+1j*bst),N-1),k=1)+np.diag(np.repeat(j*np.exp(-1j*A-1j*bst),N-1),k=-1)        
    if loc=='left': #local potential on the left edge
        kine[:2,:2]=kine[:2,:2]+np.diag(np.repeat(-lp,2))
    if loc=='both': #on both edges
        kine[:5,:5]=kine[:5,:5]+np.diag(np.repeat(-lp,5))
        kine[-5:,-5:]=kine[-5:,-5:]-np.diag(np.repeat(-lp,5))
    # if dis[0,0]: #disorder
    #     kine=kine+dis
    sorb=np.diag(-alpha*np.exp(1j*A+1j*bst),k=1)+np.diag(alpha*np.exp(-1j*A-1j*bst),k=-1)
    # sorb=np.diag(np.repeat(-alpha*np.exp(1j*A+1j*bst),N-1),k=1)+np.diag(np.repeat(alpha*np.exp(-1j*A-1j*bst),N-1),k=-1)
    if BC=="P": #periodic boundary cond.
        A=0
        kine[0,-1]=j*np.exp(-1j*A-1j*bst)
        kine[-1,0]=j*np.exp(1j*A+1j*bst)
        sorb[0,-1]=alpha*np.exp(-1j*A-1j*bst)
        sorb[-1,0]=-alpha*np.exp(1j*A+1j*bst)
    ide=np.eye(N)
    zer=np.zeros((N,N))
    # return 0.5*np.block([[kine+ide*B/2,sorb,delta*ide,zer],[sorb.T.conj(),kine-ide*B/2,zer,delta*ide],[(delta).conj()*ide,zer,-kine+ide*B/2,-sorb],[zer,(delta).conj()*ide,-sorb.T.conj(),-kine-ide*B/2]])
    return 0.5*np.vstack((np.hstack((kine+ide*B/2,sorb,delta*ide,zer)),np.hstack((sorb.T.conj(),kine-ide*B/2,zer,delta*ide)),np.hstack(((delta).conj()*ide,zer,-kine+ide*B/2,-sorb)),np.hstack((zer,(delta).conj()*ide,-sorb.T.conj(),-kine-ide*B/2))))

@njit
def DISO(N,W,shape="B"): #disorder (random box or periodic)
    if shape=="B":
        return np.diag(np.random.uniform(-W/2,W/2,N))
    if shape=="P":
        return np.diag(W*np.cos(np.arange(N)))


