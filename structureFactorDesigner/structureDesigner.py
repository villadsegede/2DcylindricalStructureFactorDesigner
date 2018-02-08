from __future__ import division
import pylab as pl
from matplotlib.ticker import MaxNLocator
from scipy.optimize import fmin_l_bfgs_b

from cylinderEnsemble import cylinderEnsemble
#from mmaClass import mmaClass	#This library is not publicly available
from powertools import outputFolder

_FD = 0
_collisionConstraint = 0
_show = 0
_useMMA = False	#True: MMA, False: BFGS
movelim = 0.04	#40 nm should be enough..
NcollisionRuns = 4

def main(FOUT):
	seed = int(pl.rand()*1e8)
	OUT = outputFolder(FOUT)
	print("Seed: {:}\t\t(now this is actually correctly implemented)".format(seed))
	pl.seed(seed)
	maxiter = 70
	Nk = 101
	bbox = (-10,10)
	Nphi = 91
	itshift = 0
	r = 0.25
	fillFactor = 1.05

	k0 = 2*pl.pi/(2.1*r)

	klim = (1.,6/4*k0)
	ce = cylinderEnsemble(klim,Nk,Nphi,bbox)
	sigmaK =   0.2
	sigmaPhi = 6/180*pl.pi
	
	#Hex
	myS = sum([gaussPeak(ce,k0,phi0/180*pl.pi,sigmaK,sigmaPhi) for phi0 in [0,60,120,180,240,300]])
	#Rect (excluding some corners)
	#myS = sum([gaussPeak(ce,k0,phi0/180*pl.pi,sigmaK,sigmaPhi) for phi0 in [0,90,180,270]])
	#Uniform
	#myS = gaussPeak(ce,k0,0/180*pl.pi,sigmaK,2*pl.pi)

	#Funky
	#myS = gaussPeak(ce,k0,45/180*pl.pi,sigmaK,0.1*pl.pi)
	#myS += gaussPeak(ce,k0,225/180*pl.pi,sigmaK,0.1*pl.pi)

	#One only
	#myS = gaussPeak(ce,k0,90/180*pl.pi,0.5*0.3,2/180*pl.pi)

	Sref = myS/ce.integrateKPhi(myS)
	if 0:
		Sref /= Sref.max()
		ce.polarPlotS(Sref)
		pl.show()
		exit()


	lCyl = pl.array([ ( (pl.random()-0.5)*2*bbox[0], (pl.random()-0.5)*2*bbox[1], r) for i in range(1000)])
	ce.updateCylinders(lCyl)
	ce.removeCollisions(r)
	ce.fillUpVoids(r,r*fillFactor)

	if _show:
		ce.polarPlotS(Sref)
		pl.show()

	ce.visualise(show=False)
	pl.savefig(OUT+"start.pdf")
	if _show:
		pl.show()
	pl.close()

	#ce.polarPlot(clim=(0,0.2))

	for ii in range(NcollisionRuns):
		if ii > 0:
			ce.removeCollisions(r)
			ce.fillUpVoids(r,r*fillFactor)
			itshift += 100

		x0 = ce.getx()
		N = x0.size
		M = 1

		iterHist = []
		if _useMMA:	
			mma = mmaClass(N,M,xmin=bbox[0],xmax=bbox[1],movelim=movelim,volfrac=1.)
			xmma,f0,df0dx,f,dfdx = mma.initializeInputs()
			xmma[:] = x0
			if _FD:
				mma.FDcheck(elm=[0,1,5,6,3]);maxiter=50

			for it in range(1,maxiter):
					f0[:],df0dx[:] = interfaceCalcSAnddS(xmma,Sref,ce,plot=False,saveIter=itshift+it,iterHist=iterHist,outFolder=OUT)
					if _collisionConstraint:
						f[0],dfdx[:,0] = interfaceCollisionConstraint(xmma,ce)
					else:
						f[0],dfdx[:,0] = mma.volfrac(xmma)
					print "{:}\t{:.3f}\t\t{:.3f}".format(str(it+itshift).zfill(4),f0[0],f[0])

					xmma[:] = mma.mmasub(it,xmma,f0,df0dx,f,dfdx)
					if abs(mma.xold2-xmma).max() < 1e-3:
						break
			xres = xmma
			f = f0[0]

		else:
			xbfgs = x0+0.
			it = 1
			def _wrapperFunc(xonly):
				return interfaceCalcSAnddS(xonly,Sref,ce,plot=False,saveIter=itshift+it,iterHist=iterHist,outFolder=OUT)
			xres,f0,d = fmin_l_bfgs_b(_wrapperFunc,xbfgs,bounds=[bbox]*xbfgs.size,maxiter=maxiter,iprint=1)
			xres = xbfgs
			f = f0
				

	#print xres
	pl.save(OUT+"xres",xres)
	print "Final val:", interfaceCalcSAnddS(xres,Sref,ce,True)[0]
	ce.visualise(show=False)
	pl.savefig(OUT+"finish.pdf")
	if _show:
		pl.show()
	pl.close()
	if _show:
		ce.polarPlot(clim=(0,0.2))

	if _show:
		ce.removeCollisions(r)
		ce.visualise()
		ce.polarPlot(clim=(0,0.2))


def gaussPeak(ce,k0,phi0,sigmaK,sigmaPhi):
	tmp = ce.lphi-phi0
	tmp[tmp>pl.pi] = (tmp[tmp>pl.pi] - 2*pl.pi)
	tmp[tmp<-pl.pi] = (tmp[tmp<-pl.pi] + 2*pl.pi)
	Ktrans,Phitrans = pl.meshgrid(ce.lk-k0,tmp)

	return 1/(sigmaK*sigmaPhi*pl.sqrt(2*pl.pi))*pl.exp(-0.5 * ((Ktrans/sigmaK)**2+(Phitrans/sigmaPhi)**2))
	


def interfaceCalcSAnddS(x,Sref,ce,plot=False,saveIter=False,iterHist=[],maxIter=False,outFolder=""):
	r = ce.lCyl[0,2]
	ce.UpdateCylinderFromX(x,r)

	lk,wk = ce.lk,ce.wk #calcGauss(Nk,klim[0],klim[1])
	phi,wphi = ce.lphi,ce.wphi

	#Sref = ce.Sref

	S = ce.calcS()
	dS = ce.calcdS()

	#integration over k and phi for elements in dS
	def dintegrateKPhi(dS):
		tmp = (ce.wk[pl.newaxis].T*dS).sum(1)
		return 1/(2*pl.pi)*(ce.wphi*tmp).sum(0)

	SA = ce.integrateKPhi(S)
	dSA = dintegrateKPhi(dS)
	#return SA,dSA

	SN = S/SA
	dSN = (dS*SA- pl.swapaxes(S[pl.newaxis].T*dSA[pl.newaxis],0,1))/SA**2
	#return SN[20,30],dSN[20,30,:]

	#Miminize average difference
	cost = ce.integrateKPhi(abs(SN-Sref)**2)
	dcost = 2*dintegrateKPhi(pl.swapaxes((SN-Sref)[pl.newaxis].T,0,1)*dSN)
	#return cost,dcost

	#Minimize standard deviation
	c1 = 100
	A = ce.integrateKPhi(S*0+1)
	mu = cost/A
	dmu = dcost/A
	SD = SN-Sref
	dSD = dSN

	STD = c1*ce.integrateKPhi((SD-mu)**2)/A 
	dSTD = 2*c1*dintegrateKPhi((dSD-dmu)*pl.swapaxes((SD-mu)[pl.newaxis].T,0,1))/A
	#print "\t\t",cost

	iterHist += [STD]
	if saveIter:
		pl.close()
		#Subplots explained:
		#https://stackoverflow.com/questions/3584805/in-matplotlib-what-does-the-argument-mean-in-fig-add-subplot111
		fig = pl.figure()
		fig.add_subplot(131)
		ce.visualise(show=False)
		pl.xticks([])
		pl.yticks([])
		fig.add_subplot(132,projection='polar')
		ce.polarPlotS(S,clim=(0,0.5),colorbar=False,useAxis=pl.gca())
		pl.xticks([])
		pl.yticks([])
		fig.add_subplot(133)
		pl.plot(range(len(iterHist)),iterHist)
		pl.plot(len(iterHist)-1,iterHist[-1],'*')
		pl.ylim(0,max(iterHist))
		if maxIter:
			pl.xlim(0,maxIter)
		else:
			pl.xlim(0,len(iterHist))
		x0,x1 = pl.gca().get_xlim()
		y0,y1 = pl.gca().get_ylim()
		pl.gca().set_aspect(abs(x1-x0)/abs(y1-y0))
		pl.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
		if len(iterHist) > 2:
			pl.xticks([0,len(iterHist)-1])
		else:
			pl.xticks([0])
		#Save the figure in a parallel thread while continuing the optimization
		#(and pray that we finish before pl.close())
		#... this doesn't work. Maybe because the thread doesn't have a window manager?
		#OUT2 = "/Users/vej22/Documents/Bacteria/Modelling/TopOptStructureFactor/"+OUT
		#thread = Thread(target=fig.savefig,kwargs={'fname':OUT2+"iter"+str(saveIter).zfill(4)+".png",
		#	'bbox_inches':'tight','dpi':150})
		#thread.start()
		fig.savefig(outFolder+"iter"+str(saveIter).zfill(4)+".png",bbox_inches='tight',dpi=150)
		pl.close()
	return STD,dSTD

def interfaceCollisionConstraint(x,ce):
	ce.UpdateCylinderFromX(x,0.5)
	N = ce.Ncyl
	f = 0
	df = pl.zeros(x.size)
	d = 0.9/2	#distance between objects
				# (I have given a 0.1 slack, since heavi(0) = 0.5)
	beta = 4
	def heavi(x):
		return -0.5*(pl.tanh(beta*x)/pl.tanh(beta)-1)
	def dHeavi(x):
		return -0.5*beta*(1-pl.tanh(beta*x)**2)/pl.tanh(beta)

	if 0:
		x = pl.linspace(-3,3)
		pl.plot(x,heavi(x))
		pl.show()
		exit()

	#c = 1e-3	#Set low for FD check
	c = 1

	lCylNew = ce.lCyl
	for i in range(N):
		for j in range(i+1,N):
			dist = (lCylNew[i][0]-lCylNew[j][0])**2 + (lCylNew[i][1]-lCylNew[j][1])**2
			ftmp = c*(dist-d)
			f += heavi(ftmp)
			df[i]   += dHeavi(ftmp)*c*2*(lCylNew[i][0]-lCylNew[j][0])
			df[j]   -= dHeavi(ftmp)*c*2*(lCylNew[i][0]-lCylNew[j][0])
			df[i+N] += dHeavi(ftmp)*c*2*(lCylNew[i][1]-lCylNew[j][1])
			df[j+N] -= dHeavi(ftmp)*c*2*(lCylNew[i][1]-lCylNew[j][1])
	return f-0.01,df



if __name__ == "__main__":
	for i in range(10,21):
		main("HighlyOrdered_"+str(i))
