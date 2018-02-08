from __future__ import division
import pylab as pl
from scipy.optimize import fmin_l_bfgs_b

from cylinderEnsemble import cylinderEnsemble


def main():
	#-- Optimization parameters
	maxiter = 15
	Nk = 101
	Nphi = 91
	bbox = (-5,5)
	r = 0.25
	fillFactor = 1.05
	k0 = 2*pl.pi/(2.1*r)	#Design wave vector
	klim = (1.,6/4*k0)	#Domain of k
	#Parameters for the gaussians used to define reference pattern
	sigmaK =   5*0.2			
	sigmaPhi = 2*6/180*pl.pi

	ce = cylinderEnsemble(klim,Nk,Nphi,bbox)
	
	#Hexagonal pattern
	Sref = sum([gaussPeak(ce,k0,phi0/180*pl.pi,sigmaK,sigmaPhi) for phi0 in [0,60,120,180,240,300]])
	Sref = Sref/ce.integrateKPhi(Sref)

	print("Plotting target structure factor")
	ce.polarPlotS(Sref)
	pl.show()

	#Generating random initial guess
	lCyl = pl.array([ ( (pl.random()-0.5)*2*bbox[0], (pl.random()-0.5)*2*bbox[1], r) for i in range(200)])
	ce.updateCylinders(lCyl)	#add random initial guess to object
	ce.removeCollisions(r)		#do collision check and remove colliding cylinders
	ce.fillUpVoids(r,r*fillFactor)	#Fill of the rest of the area (allowing a bit of space in between)

	print("Plotting initial guess")
	ce.visualise(sameColour=True)

	print("Plotting initial structure factor")
	ce.polarPlot(clim="auto")

	x0 = ce.getx()	#Get a flat array copy of x- and y-coordinates for the cylinders
	xres,f0,d = fmin_l_bfgs_b(calcCostdCost,x0,args=(Sref,ce),
						bounds=[bbox]*x0.size,maxiter=maxiter,iprint=1)

	ce.removeCollisions(r)
	print("Plotting final design after removal of collisions")
	ce.visualise(sameColour=True)
	print("Plotting final structure factor")
	ce.polarPlot(clim="auto")



##This function is used to generate the reference structure factor by calculating a 2D gaussian curve
def gaussPeak(ce,k0,phi0,sigmaK,sigmaPhi):
	tmp = ce.lphi-phi0
	tmp[tmp>pl.pi] = (tmp[tmp>pl.pi] - 2*pl.pi)
	tmp[tmp<-pl.pi] = (tmp[tmp<-pl.pi] + 2*pl.pi)
	Ktrans,Phitrans = pl.meshgrid(ce.lk-k0,tmp)
	return 1/(sigmaK*sigmaPhi*pl.sqrt(2*pl.pi))*pl.exp(-0.5 * ((Ktrans/sigmaK)**2+(Phitrans/sigmaPhi)**2))
	
#Calculate cost function and its derivative
def calcCostdCost(x,Sref,ce):
	r = ce.lCyl[0,2]	#radii don't change, but positions are updated by optimization algorithm
	ce.UpdateCylinderFromX(x,r)
	S = ce.calcS()
	dS = ce.calcdS()

	#integration over k and phi for elements in dS
	def dintegrateKPhi(dS):
		lk,wk = ce.lk,ce.wk 
		phi,wphi = ce.lphi,ce.wphi
		tmp = (ce.wk[pl.newaxis].T*dS).sum(1)
		return 1/(2*pl.pi)*(ce.wphi*tmp).sum(0)

	SA = ce.integrateKPhi(S)
	dSA = dintegrateKPhi(dS)

	SN = S/SA
	dSN = (dS*SA- pl.swapaxes(S[pl.newaxis].T*dSA[pl.newaxis],0,1))/SA**2

	#Miminize average difference
	cost = ce.integrateKPhi(abs(SN-Sref)**2)
	dcost = 2*dintegrateKPhi(pl.swapaxes((SN-Sref)[pl.newaxis].T,0,1)*dSN)
	#return cost,dcost

	#Minimize standard deviation
	c1 = 1e3	#A reasonable numerical scaling to give near-unity values for the optimizer
	A = ce.integrateKPhi(S*0+1)	#inefficient, naive implementation
	mu = cost/A
	dmu = dcost/A
	SD = SN-Sref
	dSD = dSN

	STD = c1*ce.integrateKPhi((SD-mu)**2)/A 
	dSTD = 2*c1*dintegrateKPhi((dSD-dmu)*pl.swapaxes((SD-mu)[pl.newaxis].T,0,1))/A
	return STD,dSTD

if __name__ == "__main__":
	main()
