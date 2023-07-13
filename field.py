## a .py file that contains all the core of my code for the Field class object

import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import astropy.units as u
import scipy.signal as signal
import scipy.ndimage as ndi
import matplotlib

def NoneMethod(x,y):
    return 0

class Field:
    
    def __init__(self,res,dim,dimunits,fieldunits,method, bound='symm'):
        
        # Important object variables for simulation to be set here
        
        self.dim = np.array(dim)
        self.dim1 = np.linspace(dim[0][0],dim[0][1],res)*dimunits
        self.dim2 = np.linspace(dim[1][0],dim[1][1],res)*dimunits
        self.dim1v, self.dim2v = np.meshgrid(self.dim1,self.dim2)
        self.field = method(self.dim1v.value,self.dim2v.value)*fieldunits
        self.h = self.dim1[1] - self.dim1[0]
        self.mask = np.full(np.shape(self.field),True, dtype=bool)
        self.res = res
        self.dimunits = dimunits
        self.fieldunits = fieldunits
        self.mode = bound
        
    def plot(self, field, contour=False, interpol=False, vectors=False, vecdim=10, jvec=False, name="Plot"):
        
        # a plotter help function for the different fields represented in the object
        
        extent = self.dim.flatten()
        
        if interpol==False:
            inter = 'none'
        else:
            inter = interpol
        
        plt.figure(figsize=(15,15))
        plt.title(name)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        
        if jvec == False:
            plt.imshow(field.value, origin='lower', extent=extent, interpolation = inter, norm = matplotlib.colors.TwoSlopeNorm(0), cmap='rainbow')
            
            plt.colorbar()
        
        if contour==True: 
            
            plt.contour(field.value, colors='k',origin='lower', extent=extent)
        
        if vectors==True:
            
            X = np.linspace(self.dim[0][0],self.dim[0][1],vecdim+2)[1:-1]
            Y = np.linspace(self.dim[1][0],self.dim[1][1],vecdim+2)[1:-1]
            
            XYs = np.stack(np.meshgrid(X,Y),axis=-1).reshape(1,(vecdim)**2,2)[0].T
            
            Is = np.linspace(0,self.res,vecdim+2, dtype='int')[1:-1]
            Ixys = np.stack(np.meshgrid(Is,Is),axis=-1).reshape(1,(vecdim)**2,2)[0].T
            
            dX,dY = np.gradient(field.value)/self.h.value
            
            dX = dX[Ixys[0],Ixys[1]]
            dY = dY[Ixys[0],Ixys[1]]
            
            dr = np.sqrt(dX**2 + dY**2)+0.00000001
            dX = - dX / dr
            dY = - dY / dr
            
            plt.quiver(XYs[1],XYs[0], dY, dX, dr, pivot='tail', units='width', cmap='Spectral', scale=vecdim*1.2)
            
        return plt.show()
    
    def field_direct(self, fieldref):
        
        self.field = fieldref
        
        # ^This is meant to directly alter the field in case need be
    
    def addmask(self, method):
        
        self.field = method(self.dim1v.value,self.dim2v.value)*self.field.unit
        self.mask = self.field == 0
        
        # ^Used to define local constant voltages across the field by altering the mask used to process over
        
    def submask(self, method):
        
        self.mask -= method(self.dim1v.value,self.dim2v.value)
        
        # ^Used to alter the mask if need be
    
    def resetmask(self, method):
        
        self.mask = method(self.dim1v.value,self.dim2v.value)
        
        # ^Used to alter the mask if need be
    
    def laplace(self):
        
        # Calculates the Laplacian of the field using a first-order accurate matrix convolution
        
        self.lapmat = [[0,1,0],[1,-4,1],[0,1,0]] / self.h.value**2
        
        if self.mode != 'diff':
            self.lap = np.array(signal.convolve2d(self.field,self.lapmat,mode='same', boundary=self.mode))*self.fieldunits/self.dimunits**2
        else:
            expansion = self.expand(self.field)
            self.lap = np.array(signal.convolve2d(expansion,self.lapmat,mode='same', boundary='fill'))[1:-1,1:-1]*self.fieldunits/self.dimunits**2
        
    def lap_ref(self,res,dim,dimunits,fieldunits,method):
        
        self.lapref = -method(self.dim1v.value,self.dim2v.value)*self.fieldunits/self.dimunits**2
        
        # Sets the target Laplacian using a known function (e.g. charge density / electric permitivity) to solve
        
    def mask_direct(self, mask):
        
        self.mask = mask
        
        # ^Used to alter the mask if need be
        
    def lap_ref_direct(self,laprf):
        
        self.lapref = laprf
        
        # ^sets the target Laplacian to a known Laplacian in array format
        
    def rlx(self):
        
        # Computes f(x+h,y)+f(x-h,y)+f(x,y+h)+f(x,y-h) for each cell f(x,y) 
        
        self.rlxmat = [[0,1,0],[1,0,1],[0,1,0]]
        
        if self.mode != 'diff':
            self.rlex = np.array(signal.convolve2d(self.field, self.rlxmat, mode='same', boundary=self.mode))*self.fieldunits
        else:
            expansion = self.expand(self.field)
            self.rlex = np.array(signal.convolve2d(expansion, self.rlxmat, mode='same', boundary='fill'))[1:-1,1:-1]*self.fieldunits
            
    def relax(self, iterations):
        
        # Computes an entire relaxation step
        
        for i in range(iterations):
            self.rlx()
            self.result = self.rlex-self.lapref*self.h**2
            self.field[self.mask] = self.result[self.mask]/4
            
    def MultiSolve(self, depth, n, progress=False):
        
        # Down samples our field into another Field object to relax for a given number of steps and resamples until all given depths and ns are processed
        
        if type(n)==int:
            n = np.full(len(depth),n,dtype='int')
        elif len(n)!=len(depth):
            raise ValueError("Length of n does not match length of depths")
    
        lapref = ndi.zoom(self.lapref, 1/depth[0], order=0, grid_mode=True, mode='grid-constant')
        mask = ndi.zoom(self.mask, 1/depth[0], order=0, grid_mode=True, mode='grid-constant')
        fieldref = ndi.zoom(self.field, 1/depth[0], order=0, grid_mode=True, mode='grid-constant')

        F = Field(lapref.shape[0], self.dim, self.dimunits, self.fieldunits, np.vectorize(NoneMethod), bound=self.mode)

        F.field_direct(fieldref*self.field.unit)
        F.lap_ref_direct(lapref*self.lapref.unit)
        F.mask_direct(mask)
        F.relax(n[0])

        for i in range(len(depth)-1):

            if progress == True: print("Calculating Step: ", i+1)

            lapref = ndi.zoom(self.lapref, 1/depth[i+1], order=0, grid_mode=True, mode='grid-constant')
            mask = ndi.zoom(self.mask, 1/depth[i+1], order=0, grid_mode=True, mode='grid-constant')
            fieldref = ndi.zoom(F.field, mask.shape[0]/F.field.shape[0], order=2, grid_mode=True, mode='grid-constant')
            fieldorig = ndi.zoom(self.field, 1/depth[i+1], order=0, grid_mode=True, mode='grid-constant')
            fieldref[~mask] = fieldorig[~mask]

            F = Field(fieldref.shape[0], self.dim, self.dimunits, self.fieldunits, NoneMethod, bound=self.mode)
            F.field_direct(fieldref*self.field.unit)
            F.lap_ref_direct(lapref*self.lapref.unit)
            F.mask_direct(mask)
            F.relax(n[i+1])
        
        self.field[self.mask] = F.field[self.mask]
        
    def expand(self, field):
        
        # the padding scheme for the 'diff' method; uses a 1/2 damping factor when calculating the outside padding with a finite difference - WIP
        
        top1 = field[0]
        top2 = field[1]
        top3 = field[2]
        top4 = field[3]
        top0 = top1 + 1/2*(11/6*top1 - 3*top2 + 3/2*top3 - top4/3) 
        
        bot1 = field[-1]
        bot2 = field[-2]
        bot3 = field[-3]
        bot4 = field[-4]
        bot0 = bot1 + 1/2*(11/6*bot1 - 3*bot2 + 3/2*bot3 - bot4/3) 
        
        field = np.vstack((top0,field,bot0))
        
        fieldT = field.T
        
        top1 = fieldT[0]
        top2 = fieldT[1]
        top3 = fieldT[2]
        top4 = fieldT[3]
        top0 = top1 + 1/2*(11/6*top1 - 3*top2 + 3/2*top3 - top4/3) 
        
        bot1 = fieldT[-1]
        bot2 = fieldT[-2]
        bot3 = fieldT[-3]
        bot4 = fieldT[-4]
        bot0 = bot1 + 1/2*(11/6*bot1 - 3*bot2 + 3/2*bot3 - bot4/3) 
        
        fieldT = np.vstack((top0,fieldT,bot0))
        
        return fieldT.T