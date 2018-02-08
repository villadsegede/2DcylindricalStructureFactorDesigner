from __future__ import division
import pylab as pl

from cylinderEnsemble import cylinderEnsemble

DIR = ""

pl.seed(0612)
def main():
	maxiter = 50
	klim = (0.01,30)
	klim = (5,35)
	#klim = (10,30)
	Nk = 101
	bbox = (-1,10)
	Nphi = 91

	Nphi = 181
	Nk = 201

	ce = cylinderEnsemble(klim,Nk,Nphi,bbox)
	#f = open("KientzData.txt")
	f = open("KientzComplete.txt")
	bac = []
	for line in f:
		bac += [ [float(elm) for elm in line.split()] + [0.1] ]
	lCyl = pl.array(bac)

	ce.updateCylinders(lCyl)

	ce.visualise(sameColour=True,show=False)
	pl.xticks([])
	pl.yticks([])
	pl.savefig(DIR+"fullExtract.pdf",bbox_inches='tight')
	pl.show()
	ce.polarPlot(clim=(0,0.05),show=False,colorbar=False)#,smooth=0.5)
	pl.xticks([])
	pl.yticks([])
	pl.savefig(DIR+"fullPolar.png",bbox_inches='tight')
	pl.show()
	exit()

	if 1:
		xmiddle = 4
		ymiddle = 4

		idxXhalf = lCyl[:,0]>xmiddle

		lowerLeft = RemoveWhereTrue(lCyl,idxXhalf)
		idxyhalf = lowerLeft[:,1]>ymiddle
		lowerLeft = RemoveWhereTrue(lowerLeft,idxyhalf)

		lowerRight = RemoveWhereTrue(lCyl,pl.invert(idxXhalf))
		idxyhalf = lowerRight[:,1]>ymiddle
		lowerRight = RemoveWhereTrue(lowerRight,idxyhalf)

		upperLeft = RemoveWhereTrue(lCyl,idxXhalf)
		idxyhalf = upperLeft[:,1]<ymiddle
		upperLeft = RemoveWhereTrue(upperLeft,idxyhalf)

		upperRight = RemoveWhereTrue(lCyl,pl.invert(idxXhalf))
		idxyhalf = upperRight[:,1]<ymiddle
		upperRight = RemoveWhereTrue(upperRight,idxyhalf)

	ce.updateCylinders(lowerLeft)
	ce.visualise()
	ce.polarPlot(clim=(0,0.3))
	ce.updateCylinders(lowerRight)
	ce.visualise()
	ce.polarPlot(clim=(0,0.3))
	ce.updateCylinders(upperRight)
	ce.visualise()
	ce.polarPlot(clim=(0,0.3))
	ce.updateCylinders(upperLeft)
	ce.visualise()
	ce.polarPlot(clim=(0,0.3))



def RemoveWhereTrue(array,mask):
	#Find first element which is True
	idxLowest = pl.argmax(mask)
	#Find the indices that need to be removed
	idxRemoval = (pl.arange(mask.size)*mask)
	#We only want to remove the zero case if idxLowest is zero:
	idxRemoval = idxRemoval.clip(idxLowest)
	test = pl.delete(array,idxRemoval,axis=0)
	return test


if __name__ == "__main__":
	main()
