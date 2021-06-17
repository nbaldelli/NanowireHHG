# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:32:55 2021

@author: nbaldelli
"""

from joblib import Parallel, delayed
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

@njit
def NWIRE(N,j,alpha,a,B,mu,delta,A,BC="O",dis=False,loc=False): #nanowire hamiltonian
    bst=0#time reversal symmetry breaking term
    kine=np.diag(np.repeat(-mu-2*j,N))+np.diag(j*np.exp(1j*A+1j*bst),k=1)+np.diag(j*np.exp(-1j*A-1j*bst),k=-1)        
    # kine=np.diag(np.repeat(-mu-2*j,N))+np.diag(np.repeat(j*np.exp(1j*A+1j*bst),N-1),k=1)+np.diag(np.repeat(j*np.exp(-1j*A-1j*bst),N-1),k=-1)        
    if loc=='left': #local potential on the left edge
        kine[:3,:3]=kine[:3,:3]+np.diag(np.repeat(-0.1,3))
    if loc=='both': #on both edges
        kine[:3,:3]=kine[:3,:3]+np.diag(np.repeat(-0.1,3))
        kine[-3:,-3:]=kine[-3:,-3:]-np.diag(np.repeat(-0.1,3))
    if dis: #disorder
        kine=kine+DISO(N,0.001*j,shape=dis)
    sorb=np.diag(-alpha*np.exp(1j*A+1j*bst),k=1)+np.diag(alpha*np.exp(-1j*A-1j*bst),k=-1)
    # sorb=np.diag(np.repeat(-alpha*np.exp(1j*A+1j*bst),N-1),k=1)+np.diag(np.repeat(alpha*np.exp(-1j*A-1j*bst),N-1),k=-1)
    if BC=="P": #periodic boundary cond.
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


dim=15
magf=np.linspace(0.3,1.4,dim)
mus=np.linspace(-0.5,0.5,dim)
orderpa=np.zeros((dim,dim))
gaps1=np.zeros((dim,dim))
gaps2=np.zeros((dim,dim))
toc=time.time()
#%%
for i in range(len(magf)):
    for jj in range(len(mus)):
        #%%
        jj=8
        i=10
        #parameters inizialization
        N=50 #number of lattice sites
        a=1 #lattice spacing
        j=-0.3 #hopping
        alpha=0.4#spin-orbit
        delta=0.4 #superconducting coupling (initial guess)
        mu=mus[jj] #chem. potential
        B=magf[i] #mag. field (0.15 trivial, 0.7 topological)
        BC="O" #Boundary conditions ("P" or "O")
        SE='bulk' #spacial envelope
        disorder=False #toggles disorder
        TDdelta=False #toggles time dependent delta
            
        Eso=(alpha**2)/(4*j) #spin orbit coupling energy
        topinv=np.sign(delta**2+mu**2-B**2) #topological invariant
        
        omega = 0.0075  # carrier frequency
        I = 0.5 # e.m. intensity
        tmax = 2*np.pi*5/omega  # time to evolve (function of pulse period)
        fr = int(5*tmax)  # num. of frames
        dt = tmax/fr  # time step for evolution
        ax=np.linspace(0,tmax,int(tmax/dt)) #time axis
        scc=np.zeros((N,len(ax)+1),dtype=np.complex128) #superconducting coupling
        scc[:,0]=np.array([delta]*N)
        
        ##############################################################################
        #EXACT DIAGONALIZATION
        ham=NWIRE(N,j,alpha,a,B,mu,scc[:,0].conj(),np.zeros(N-1),BC=BC,dis=disorder)
        vals,vecs=alg.eigh(ham)
        
        if TDdelta==True:
            count=0; temp=0;
            scc[:,0]=np.sum(vecs[:N,:N].conj()*vecs[2*N:3*N,:N]+vecs[:N,N:2*N].conj()*vecs[2*N:3*N,N:2*N],axis=1)
            while (np.abs(vals[2*N+1]/omega-temp)>=10e-7):
                temp=vals[2*N+1]/omega
                tempvecs=vecs
                ham=NWIRE(N,j,alpha,a,B,mu,scc[:,0].conj(),0,BC=BC,dis=disorder)
                vals,vecs=alg.eigh(ham)
                scc[:,0]=-np.sum(vecs[:N,:N].conj()*vecs[2*N:3*N,:N]+vecs[:N,N:2*N].conj()*vecs[2*N:3*N,N:2*N],axis=1)
                count=count+1
                print(temp)
                if (count>=300):
                    print("Convergence error!")
                    break
        if TDdelta==False:
            scc[:,0]=np.array([delta]*N)
        
        print("Topinv=",topinv)
        print("Gap=",(vals[2*N+1])/omega)
        print("Split ratio=",vals[2*N+1]/vals[2*N])
        print("Energy radius=",(np.max(vals)-np.min(vals))/omega)
        print(B,">>",np.abs(Eso),",",delta)
        gaps1[i,jj]=(vals[2*N]); gaps2[i,jj]=(vals[2*N+1])
        # ##############################################################################
        #PLOT OF SPECTRUM AND GAP
        
        if (BC=="P"):
            k=np.zeros(4*N)
            for jj in range(len(k)):
                inn=np.where(vecs[:,jj]!=0)[0][0]
                k[jj]=np.real(1j*np.log(vecs[inn,jj]/vecs[inn+1,jj]))
            plt.figure(0,dpi=220); plt.grid(); plt.title("Spectrum") #plt.xlim(-10,10); plt.ylim(-0.5,0.5)
            plt.plot(k,vals)
        if (BC=="O"):
            plt.figure(0,dpi=220); plt.grid(); plt.title("Spectrum") #plt.xlim(-10,10); plt.ylim(-0.5,0.5)
            plt.scatter(np.linspace(0,4*N,4*N),vals)
            plt.xlim(int(2*N-N/2),int(2*N+N/2))
            plt.ylim(-0.2,0.2)
          
        # asc=np.linspace(-(N-1)*a/2,(N-1)*a/2,N)
        asc=np.linspace(0,(N-1)*a,N)
        dip=np.zeros((N,len(ax)),dtype=np.complex128)
        usq=vecs.conj()*vecs
        dip[:,0]=asc*np.sum(usq[:N,:2*N]+usq[N:2*N,:2*N],axis=1)/np.sum(usq[:N,:2*N]+usq[N:2*N,:2*N])
        # dip[:,0]=asc*usq[2*N:3*N,0]
        # plt.figure(2
        # plt.plot(asc,dip[:,0]/asc)

        if SE=='bulk':
            spenv=(np.exp(-((asc-asc[int((N-1)/2)])/(0.1*N))**2)[:-1]/sum(np.exp(-((asc-asc[int((N-1)/2)])/(0.1*N))**2)[:-1]))*(N-1)
        if SE=='edge':
            spenv=(np.exp(-((asc-asc[5])/(0.1*N))**2)[:-1]/sum(np.exp(-((asc-asc[0])/(0.1*N))**2)[:-1]))*(N-1)
        else:
            spenv=np.ones(N-1)
        ##############################################################################
        #TIME EVOLUTION
        for ind in range(1,len(ax)):
            print('step: ',ind)
            A=PULSE(ax[ind-1],I,omega,env="S",ncyc=5)*spenv
            ham=NWIRE(N,j,alpha,a,B,mu,scc[:,ind-1].conj(),A,BC=BC,dis=disorder)
            vecs=CRNI(vecs,ham,dt)
            # vecs=LANC(vecs,ham,5,dt)
            usq=vecs.conj()*vecs
            if TDdelta==True:
                scc[:,ind]=-np.sum(vecs[:N,:2*N].conj()*vecs[2*N:3*N,:2*N],axis=1)
            if TDdelta==False:
                scc[:,ind]=np.array([delta]*N)
            dip[:,ind]=asc*np.sum(usq[:N,:2*N]+usq[N:2*N,:2*N],axis=1)/np.sum(usq[:N,:2*N]+usq[N:2*N,:2*N])
            # dip[:,ind]=asc*usq[2*N:3*N,0]
            # plt.figure(44,dpi=220)
            # if (ind%300==0):
            #     plt.plot(asc,dip[:,ind]/asc)
            #     plt.show()
           
        ##############################################################################
        #FOURIER TRANSFORM 
        curr=np.gradient(np.sum(dip[:,:],axis=0),dt)
        plt.figure(5,dpi=220); plt.title("Current")
        plt.plot(ax,curr)
        plt.grid()
        norm=np.max(np.abs(np.fft.fft(EPULSE(ax,I,omega,env="S",ncyc=5))*dt)**2)
        tr = np.abs(np.fft.fft(curr)*dt)**2
        freqs = np.fft.fftfreq(fr)*fr*2*np.pi/tmax
        plt.figure(6, dpi=220); plt.title("FFT, I=1., omega=0.0075, B=1.2")
        plt.plot(freqs[:int(fr/2)]/omega, tr[:int(fr/2)]/norm)
        plt.xlim(0,100)
        plt.vlines(2*(vals[2*N+1])/omega,10e-10,10e-1)
        plt.vlines((vals[2*N+1])/omega,10e-13,10e-1)
        plt.yscale("log")
        plt.xlabel("Harmonic order")
        plt.ylabel("Emission")
        plt.grid()
        print("mu,B=",mu,B)
        print('\007')
        # np.savez_compressed("prova15"+str(B)+str(mu),np.real(dip))
#%%
for jj in range(len(mus)):
    for i in range(len(magf)):
        # jj=7
        # i=13
        mu=mus[jj]
        B=magf[i]
        # if (jj==12)and(i==1):
        #     continue
        # with np.load("prova15"+str(B)+str(mu)+".npz") as f:
        #     dip=np.real(f['arr_0'])
        
        # plt.figure(1,dpi=220); plt.title("Time dependent dipole")
        # plt.plot(ax,np.sum(dip,axis=0))
        
        # plt.figure(2,dpi=220); plt.title("Time dependent superconducting gap")
        # plt.plot(ax,np.sum(np.abs(scc[:,:-1])/N,axis=0))
        # plt.xlim(0,2)
        #############################################################################
        #FOURIER TRANSFORM
        # plt.figure(8,dpi=220)
        # plt.plot(ax,EPULSE(ax,I,omega,env="S",ncyc=5))
        # plt.xlabel("Time")
        # plt.ylabel("EM field")
        # plt.figure(9,dpi=220)
        # plt.plot(ax,np.sum(dip,axis=0))
        # curr=np.gradient(np.sum(dip,axis=0),dt)
        # plt.xlabel("Time")
        # plt.ylabel("Dipole")
        # plt.figure(10,dpi=220); plt.title("Current")
        # plt.plot(ax,curr)
        # plt.xlabel("Time")
        # plt.ylabel("Current")
       
        # # plt.grid()

        # norm=np.max(np.abs(np.fft.fft(EPULSE(ax,I,omega,env="S",ncyc=5))*dt)**2)
        # tr = np.abs(np.fft.fft(curr)*dt)**2
        # freqs = np.fft.fftfreq(fr)*fr*2*np.pi/tmax
        # plt.figure(i, dpi=220); plt.title("FFT, I="+str(I)+", omega=0.0075, B="+str(B)+",   mu="+str(mu))
        # plt.plot(freqs[:int(fr/2)]/omega, tr[:int(fr/2)]/norm)
        # plt.xlim(0,200)
        # plt.yscale("log")
        # plt.xlim(0,100)
        # plt.xlabel("Harmonic order"); plt.ylabel("Emission"); plt.grid()
        print("mu,B=",mu,B)
        eps=5
        if (gaps1[i,jj]<10e-3):
            plt.vlines(2*gaps2[i,jj]/omega,10e-10,10e-1)
            plt.vlines(gaps2[i,jj]/omega,10e-13,10e-1)
            bandgap=np.mean((tr[:int(fr/2)]/norm)[np.abs(freqs[:int(fr/2)]-2*gaps2[i,jj]).argmin()-eps:np.abs(freqs[:int(fr/2)]-2*gaps2[i,jj]).argmin()+eps+1])
            halfband=np.mean((tr[:int(fr/2)]/norm)[np.abs(freqs[:int(fr/2)]-gaps2[i,jj]).argmin()-eps:np.abs(freqs[:int(fr/2)]-gaps2[i,jj]).argmin()+eps+1])
        else:
            plt.vlines(2*gaps1[i,jj]/omega,10e-13,10)
            plt.vlines(gaps1[i,jj]/omega,10e-13,10)
            bandgap=np.mean((tr[:int(fr/2)]/norm)[np.abs(freqs[:int(fr/2)]-2*gaps1[i,jj]).argmin()-eps:np.abs(freqs[:int(fr/2)]-2*gaps1[i,jj]).argmin()+eps+1])
            halfband=np.mean((tr[:int(fr/2)]/norm)[np.abs(freqs[:int(fr/2)]-gaps1[i,jj]).argmin()-eps:np.abs(freqs[:int(fr/2)]-gaps1[i,jj]).argmin()+eps+1])
        print("Order parameter=",halfband/bandgap)
        
        if gaps1[i,jj]>10e-3:
            orderpa[jj,i]=0 #topological invariant
        else:
            orderpa[jj,i]=1
        
        # plt.figure(i,dpi=220)
        # plt.scatter(magf,orderpa[jj,:]); #plt.ylim(0,3)
        # plt.ylim(0,1)

macord=np.meshgrid(magf,mus)
plt.figure(71,dpi=220,figsize=(5,5)); plt.title("Order parameter (presence of edge states), I=1.5"); plt.xlabel("B"); plt.ylabel("mu")
aa=plt.pcolormesh(macord[0],macord[1],orderpa)
plt.colorbar(aa)
    
print(time.time()-toc)
# plt.figure(33,dpi=220)
# bb=np.array([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.1,1.2,1.3])
# op=np.array([1113,6695,269,20.85,3.64,0.1388,0.844,1.932,0.7,1.9,9.2])
# plt.scatter(bb,op)
# plt.grid(); plt.yscale("log")
# plt.xlabel("Magnetic field"); plt.ylabel("Intensity ratio")
# plt.title("Top. phase transition (mu=0, delta,alpha=0.4, j=0.3")

