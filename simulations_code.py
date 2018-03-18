#from abc import ABCMeta, abstractmethod
from matplotlib import pyplot as pl
import numpy as np
import math
import random
import scipy.stats as st
import scipy.io
#import copy

def get_p_val(value,val_list):
    #returns the p value of value compared to val list
    dataMean = np.nanmean(val_list)
    dataStd = np.nanstd(val_list)
    if dataStd == 0:  # not to allow division by 0
        return 10**(-100)

    zScore = (value-dataMean)/dataStd
    return 1-st.norm.cdf(zScore)


def rotate(l, n):
    return l[-n:] + l[:-n]


def worm_fight_lin_grad(memoryLen,pTurnNegGrad,pTurnCoeff,numberOfWorms,numberOfSteps):
    #compares the performane of a first derivative adaptation worm and a biased random walk worm on a linear gradient
    #number of steps defines the run time of each worm. the turn rate of each biased random walk worm is determined by the empiric turn rate of the parallel first derivative adaptation worm
    wormGrad = LinearGradient(1, 0)
    startLoc = Loc(0, 0)
    memoryWormsTotX = 0
    regWormsTotX = 0
    totalTurnRates = 0
    for w in range(numberOfWorms):
        thisStartOrientation = (random.random()-0.5)*2*math.pi
        thisMemoryWorm = MemoryBiasedRandomWorm(wormGrad, memoryLen, pTurnNegGrad, startLoc, pTurnCoeff,thisStartOrientation)
        for runs in range(numberOfSteps):
            thisMemoryWorm.make_step()
        thisWormPosGradTurnProb = thisMemoryWorm.get_positiveGradTurnRate()
        totalTurnRates += thisWormPosGradTurnProb
        thisRegWorm = BiasedRandomWorm(wormGrad, thisWormPosGradTurnProb, pTurnNegGrad, startLoc,thisStartOrientation)
        for runs in range(numberOfSteps):
            thisRegWorm.make_step()
        memoryWormsTotX += thisMemoryWorm.get_final_loc().get_x()
        regWormsTotX += thisRegWorm.get_final_loc().get_x()
    return memoryWormsTotX / (numberOfWorms * numberOfSteps), regWormsTotX/(numberOfWorms*numberOfSteps),totalTurnRates/numberOfWorms


def worm_fight_gaussian_grad(pTurnNegGrad,pTurnCoeff,numberOfWorms,memoryLen=30, noise=0):
    # compares the performance of a first derivative adaptation worm and a biased random walk worm on a gaussian gradient
    # number of steps defines the run time of each worm. the turn rate of each biased random walk worm is determined by the empiric turn rate of the parallel first derivative adaptation worm
    boundsR = 30
    wormGrad = GaussianGradient(100,noise)
    startX  = 300
    startLoc = Loc(startX,0)
    memoryWormsOnGradProjection = 0
    regWormsOnGradProjection = 0
    memoryWormsTotStepNum = 0
    regWormsTotStepNum = 0
    totalTurnRates = 0
    maxStepNum = startX*10
    for w in range(numberOfWorms):
        thisWormProjection = 0
        thisWormStepNum = 0
        thisStartOrientation = (random.random()-0.5)*2*math.pi
        thisMemoryWorm = MemoryBiasedRandomWorm(wormGrad, memoryLen, pTurnNegGrad, startLoc, pTurnCoeff,thisStartOrientation)
        targetReached = False
        while  targetReached == False:
            thisMemoryWorm.make_step()
            thisDirection = thisMemoryWorm.get_final_grad_phi()
            thisLoc = thisMemoryWorm.get_final_loc()
            thisWormProjection += np.cos(thisDirection-wormGrad.get_gradient_dierection(thisLoc).get_phi())
            memoryWormsTotStepNum+=1
            thisWormStepNum += 1


            if (thisLoc.get_r() < boundsR) | (thisWormStepNum > maxStepNum):
                targetReached = True
                #if thisLoc.get_r() < boundsR:
                #    print('target reached - memworm after ' + str(thisWormStepNum))
                #if thisWormStepNum > maxStepNum:
                #    print('target NOT reached - memworm')


        thisWormMeanProjection = thisWormProjection/thisWormStepNum
        memoryWormsOnGradProjection += thisWormMeanProjection/numberOfWorms
        thisWormPosGradTurnProb = thisMemoryWorm.get_positiveGradTurnRate()
        totalTurnRates += thisWormPosGradTurnProb


        thisRegWorm = BiasedRandomWorm(wormGrad, thisWormPosGradTurnProb, pTurnNegGrad, startLoc,thisStartOrientation)
        targetReached = False
        thisWormProjection = 0
        thisWormStepNum = 0
        while targetReached == False:
            thisRegWorm.make_step()
            thisDirection = thisRegWorm.get_final_grad_phi()
            thisLoc = thisRegWorm.get_final_loc()

            thisWormProjection += np.cos(thisDirection - wormGrad.get_gradient_dierection(thisLoc).get_phi())
            regWormsTotStepNum += 1
            thisWormStepNum+=1
            if (thisLoc.get_r() < boundsR) | (thisWormStepNum > maxStepNum):
                targetReached = True
                #if thisLoc.get_r() < boundsR:
                #    print('target reached - regworm after' + str(thisWormStepNum))
                #if thisWormStepNum > maxStepNum:
                #    print("target NOT reached - regworm")


        thisWormMeanProjection = thisWormProjection / thisWormStepNum
        regWormsOnGradProjection += thisWormMeanProjection/numberOfWorms
        #print(w)

    regWormMeanTimeToTarget = regWormsTotStepNum/numberOfWorms
    memoryWormMeanTimeToTarget = memoryWormsTotStepNum/numberOfWorms

    memoryWormsMeanProjection = memoryWormsOnGradProjection
    regWormsMeanProjection = regWormsOnGradProjection
    return memoryWormMeanTimeToTarget,regWormMeanTimeToTarget,memoryWormsMeanProjection,regWormsMeanProjection,totalTurnRates/numberOfWorms





class Loc():
    #defines a 2D vector
    def __init__(self,cor1,cor2,method='cart'):
        # if method is cart, cor1=x, cor2=y. if not, cor1=r, cor2 = phi
        if method == 'cart':
            self.x = cor1
            self.y = cor2
            self.r = np.sqrt(self.x**2 + self.y**2)
            self.phi = np.arctan2(self.y, self.x)

        else:
            self.r = cor1
            self.phi = cor2
            self.x = self.r * np.cos(self.phi)
            self.y = self.r * np.sin(self.phi)


    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_r(self):
        return(self.r)

    def get_phi(self):
        return(self.phi)

    def __add__(self,other):
        added_x = self.x+other.get_x()
        added_y = self.y+other.get_y()
        return Loc(added_x,added_y)

    def __sub__(self, other):
        subs_x = self.get_x()-other.get_x()
        subs_y = self.get_y()-other.get_y()
        return Loc(subs_x,subs_y)


class Gradient():
    # this is an abstract gradient class.
    def get_grad_val(self,location):
        pass

    def plot_grid_val(self,xStart,xEnd,yStart,yEnd):
        res = 100
        xrange = xEnd - xStart
        yrange = yEnd - yStart

        xsteps = np.linspace(xStart,xEnd,res)
        ysteps = np.linspace(yStart,yEnd,res)
        thisIm = np.zeros((len(xsteps),len(ysteps)))
        for i in range(len(xsteps)):
            for j in range(len(ysteps)):
                thisIm[i,j] = self.get_grad_val(Loc(xsteps[i], ysteps[j]))
        pl.figure(1)
        pl.imshow(thisIm, extent=[0, 1, 0, 1], interpolation='nearest', aspect=0.15)



class LinearGradient(Gradient):
    #creates a linear gradient that is 0 at point 0,0 and changes by dx and dy
    def __init__(self,dx,dy):
        self.dx = dx
        self.dy = dy

    def get_grad_val (self,location):
        return (self.dx*location.get_x()+self.dy*location.get_y())

    def get_gradient_dierection(self,location):
        normFac = np.sqrt(dx ** 2 + dy ** 2)
        return Loc(dx/normFac, dy/normFac)

class GaussianGradient(Gradient):
    #creates an exponential gradient that peaks to 0 at (0,0) and decays as exp(-r/std)
    def __init__(self,gaussian_std,noise=0):
        self.std = gaussian_std
        self.noise = noise
    def get_grad_val (self,location):
        dr = location.get_r()
        real_val = np.exp(-1*dr**2/(2*self.std**2))
        added_noise = np.random.normal(0, self.noise)

        return (real_val+added_noise)
    def get_gradient_dierection(self,location):
        gradX = location.get_x()
        gradY = location.get_y()
        gradR = location.get_r()
        xVec = -gradX * np.exp(-1 * gradR ** 2 / (2 * self.std ** 2)) / (self.std ** 2)
        yVec = -gradY * np.exp(-1 * gradR ** 2 / (2 * self.std ** 2)) / (self.std ** 2)
        normFac = np.sqrt(xVec**2+yVec**2)
        return Loc(xVec/normFac, yVec/normFac)




class Worm():
    # abstract class of a worm
    def __init__(self,startLoc,startOrientation,speed,wormGrad):
        self.startLoc = startLoc
        self.orientation = startOrientation
        self.speed = speed
        self.wormGrad = wormGrad
        self.allLocs = [startLoc,startLoc,startLoc]
        self.feltGrads = [wormGrad.get_grad_val(startLoc),wormGrad.get_grad_val(startLoc),wormGrad.get_grad_val(startLoc)]


    def make_step(self):
        #makes a single step of the worm in which it should advance by 1 speed units
        pass

    def get_allLocs(self):
        #returns all the positions of the worm so far
        return(self.allLocs)

    def get_final_loc(self):
        #returns just the final location of the worm
        return self.allLocs[-1]

    def get_final_grad_phi(self):
        #returns the last orientation of the worm
            return (self.allLocs[-1]-self.allLocs[-2]).get_phi()

    def get_gradient(self):
        #returns the gradient that defines the worms arena
        return self.wormGrad

    def plot_track(self,fig=1):
        #plots the tracl of the worm
        all_locs = self.get_allLocs()
        all_xs = map(lambda x: x.get_x(), all_locs)
        all_ys = map(lambda x: x.get_y(), all_locs)
        pl.figure(fig)
        pl.plot(all_xs,all_ys)
        pl.axes().set_aspect('equal', 'datalim')

    def save_track(self,saveLoc):
        all_locs = self.get_allLocs()
        all_xs = map(lambda x: x.get_x(), all_locs)
        all_ys = map(lambda x: x.get_y(), all_locs)
        resDict = {}
        resDict['X'] = all_xs
        resDict['Y'] = all_ys
        scipy.io.savemat(saveLoc, resDict)

class BiasedRandomWorm(Worm):
    #implements a biased random walk entity. only considers the sign of the first derivative. in a positive gradient will turn with probability pTurnPos
    # in negative gradient will turn with probability pNegTurn
    def __init__(self,wormGrad,pTurnPosGrad,pTurnNegGrad,startLoc,startOrientation=0,speed=1,):
        Worm.__init__(self,startLoc,startOrientation,speed,wormGrad)
        self.pTurnPosGrad = pTurnPosGrad
        self.pTurnNegGrad = pTurnNegGrad

    def set_p_turn_pos_grad(self,pTurnPosGrad):
        self.pTurnPosGrad = pTurnPosGrad

    def make_step(self):
        #make a step for the worm
        first_der = self.feltGrads[-1]-self.feltGrads[-2]  #compute the first derivative
        if first_der < 0:  # if negative first derivative
            if random.random() < self.pTurnNegGrad:
                turn = 1
            else:
                turn = 0
        else:  #if positive first derivative
            if random.random() < self.pTurnPosGrad:
                turn = 1
            else:
                turn = 0

        if turn == 1:  #if drawed a turn, choose new orientation randomly
            self.orientation = (random.random()-0.5)*2*math.pi

        dLoc = Loc(self.speed, self.orientation, 'pol')  #defines the change in location
        self.allLocs.append(dLoc+self.allLocs[-1])
        self.feltGrads.append(self.wormGrad.get_grad_val(self.allLocs[-1]))  #add the concentration at the new location to the list

    def get_positiveGradTurnRate(self):
        return self.pTurnPosGrad



class MemoryBiasedRandomWorm(Worm):
    def __init__(self,wormGrad,memoryLen,pTurnNegGrad,startLoc,pTurnCoeff=1,startOrientation=0,speed=1,):
        Worm.__init__(self, startLoc, startOrientation, speed, wormGrad)
        self.pTurnCoeff = pTurnCoeff
        self.pTurnNegGrad = pTurnNegGrad
        self.memoryLen = memoryLen
        self.allFirstDers = [np.nan]*(memoryLen -1)+ [0]
        self.positiveGradTurns = 0
        self.positiveGradMoves = 0


    def get_positiveGradTurnRate(self):
        return self.positiveGradTurns/float(self.positiveGradMoves)

    def make_step(self):
        first_der = self.feltGrads[-1]-self.feltGrads[-2]
        pVal = get_p_val(first_der,self.allFirstDers)

        if first_der<0:
            if random.random()<self.pTurnNegGrad:
                turn = 1
            else:
                turn = 0
        else:
            pTurn = pVal*self.pTurnCoeff
            if random.random()<pTurn:
                turn = 1
                self.positiveGradTurns+=1
            else:
                turn = 0
            self.positiveGradMoves+=1
        if turn==1:
            self.orientation = (random.random()-0.5)*2*math.pi

        dLoc = Loc(self.speed,self.orientation,'pol')
        self.allLocs.append(dLoc+self.allLocs[-1])
        self.feltGrads.append(self.wormGrad.get_grad_val(self.allLocs[-1]))
        self.allFirstDers = rotate(self.allFirstDers, 1)
        self.allFirstDers[0] = first_der

def linearGradParamScan():
    allpTurnNegGrads = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    allPTurnCoeff = [0.001, 0.01, 0.02, 0.03, 0.04, 0.08, 0.16, 0.24, 0.32, 0.48, 0.64, 0.82, 1]
    memoryLen = 50
    numberOfWorms = 10
    numberOfSteps = 1000
    memoryWormRates = np.zeros((len(allpTurnNegGrads),len(allPTurnCoeff)))
    regularWormRates = np.zeros((len(allpTurnNegGrads),len(allPTurnCoeff)))
    empiricTurnProbs = np.zeros((len(allpTurnNegGrads),len(allPTurnCoeff)))
    saveLoc = 'Z:/shared_lab/pythonCode/linearData.mat'
    for i in range(len(allpTurnNegGrads)):
        for j in range(len(allPTurnCoeff)):
            thisRes = worm_fight_lin_grad(memoryLen, allpTurnNegGrads[i], allPTurnCoeff[j], numberOfWorms, numberOfSteps)
            memoryWormRates[i,j] = thisRes[0]
            regularWormRates[i,j] = thisRes[1]
            empiricTurnProbs[i,j] = thisRes[2]
            print (i, j)

    resDict = {}
    resDict['memoryWormMeanProjection'] = memoryWormRates
    resDict['regularWormMeanProjection'] = regularWormRates
    resDict['empiricPosTurnRate'] = empiricTurnProbs
    resDict['wormNum'] = numberOfWorms
    resDict['allpTurnNegGrads'] = allpTurnNegGrads
    resDict['allPTurnCoeff'] = allPTurnCoeff
    resDict['memoryLen'] = memoryLen
    resDict['numberOfWorms'] = numberOfWorms
    scipy.io.savemat(saveLoc, resDict)



    scipy.io.savemat(saveLoc, resDict)

    return [memoryWormRates,regularWormRates,empiricTurnProbs]

def gaussianSingleWormProjectionsAnalysis(Worm,boundsR,doPlot=False):
    rs = []
    projs = []
    targetReached =False
    while  targetReached == False:
        Worm.make_step()
        thisDirection = Worm.get_final_grad_phi()
        thisLoc = Worm.get_final_loc()
        projs.append(np.cos(thisDirection-Worm.get_gradient().get_gradient_dierection(thisLoc).get_phi()))
        rs.append(thisLoc.get_r())
        if thisLoc.get_r() < boundsR:
               targetReached=True
    if doPlot:
        pl.plot(rs, projs,'.')
        pl.show()
    return np.asarray(rs), np.asanyarray(projs)





def gaussianManyWormsProjectionAnalysis(gradStd,pTurnNegGrad,pTurnCoeff,numberOfWorms,memoryLength=50,boundsR=20,startLocX=-2000):
    saveLoc = 'C:/rotem/GaussianDataRProjections.mat'

    thisGrad =GaussianGradient(gradStd)
    meanRes = 20
    allRs = range(0,abs(startLocX),meanRes)
    allMemRsCounts = np.zeros((len(allRs),1))
    projsMemSum = np.zeros((len(allRs),1))

    allRegRsCounts = np.zeros((len(allRs),1))
    projsRegSum = np.zeros((len(allRs),1))

    startLoc = Loc(startLocX,0)
    for w in xrange(numberOfWorms):
        thisStartOrientation = (random.random() - 0.5) * 2 * math.pi
        thisMemoryWorm = MemoryBiasedRandomWorm(thisGrad,memoryLength,pTurnNegGrad,startLoc,pTurnCoeff,thisStartOrientation)

        thisMemResult = gaussianSingleWormProjectionsAnalysis(thisMemoryWorm,boundsR)
        thisPosRate = thisMemoryWorm.get_positiveGradTurnRate()
        thisRegWorm = BiasedRandomWorm(thisGrad,thisPosRate,pTurnNegGrad,startLoc,thisStartOrientation)
        thisRegResult = gaussianSingleWormProjectionsAnalysis(thisRegWorm,boundsR)

        rsReg = thisRegResult[0]
        rsReg = np.round(rsReg / meanRes) * meanRes
        ProjsReg = thisRegResult[1]

        rsMem = thisMemResult[0]
        rsMem = np.round(rsMem / meanRes) * meanRes
        ProjsMem = thisMemResult[1]
        for i in xrange(len(allRs)):
            thisRegSum = np.sum(ProjsReg[rsReg==allRs[i]])
            thisRegCount = np.count_nonzero(rsReg==allRs[i])
            allRegRsCounts[i]+=thisRegCount
            projsRegSum[i]+=thisRegSum

            thisMemSum = np.sum(ProjsMem[rsMem == allRs[i]])
            thisMemCount = np.count_nonzero(rsMem == allRs[i])
            allMemRsCounts[i] += thisMemCount
            projsMemSum[i] += thisMemSum
        print(w)

    resDict = {}
    resDict['empiricPosTurnRate'] = thisPosRate
    resDict['allRs'] = allRs
    resDict['memMeanProjections'] = projsMemSum/allMemRsCounts
    resDict['regMeanProjections'] = projsRegSum/allRegRsCounts
    resDict['gradSTD'] = gradStd
    resDict['boundsR'] = boundsR

    scipy.io.savemat(saveLoc, resDict)
   # pl.plot(allRs,projsMemSum/allMemRsCounts)
    #pl.show()




def gaussianGradParamScan(noise_level,saveLoc = 'c:\\Temp'):
    allpTurnNegGrads = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.85, 0.9, 0.95,0.97,  1]
    #allpTurnNegGrads = [0.1, 0.3, 0.6, 0.9]
    allPTurnCoeff = [0.00001,0.0001, 0.001, 0.01, 0.02,0.03, 0.04, 0.08, 0.16, 0.24, 0.27, 0.32, 0.4, 0.44, 0.48, 0.52, 0.56, 0.6, 0.64, 0.68, 0.73, 0.78, 0.82, 0.86, 0.91, 0.96, 1, 1.12, 1.25, 1.5, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 5096,10000]

    #allPTurnCoeff = [40000,10000,2048,512,128,64,32,16,8,4,2,1.5,1.25,1.12,1,0.96,0.91,0.86,0.82,0.78,0.73,0.68,0.64,0.6,0.56,0.52,0.48,0.44,0.4,0.32,0.27,0.24,0.16,0.08,0.04,0.03,0.02,0.01,0.001]
    allPTurnCoeff = [10, 5, 2, 1, 0.5, 0.1]

    numberOfWorms = 400
    memory_len = 30
    #numberOfWorms = 3
    memoryWormMeanProjection = np.zeros((len(allpTurnNegGrads), len(allPTurnCoeff)))
    memoryWormStepNum = np.zeros((len(allpTurnNegGrads), len(allPTurnCoeff)))

    regularWormMeanProjection = np.zeros((len(allpTurnNegGrads), len(allPTurnCoeff)))
    regularWormStepNum = np.zeros((len(allpTurnNegGrads), len(allPTurnCoeff)))

    regularWormStepNum = np.zeros((len(allpTurnNegGrads), len(allPTurnCoeff)))
    empiricTurnProbs = np.zeros((len(allpTurnNegGrads), len(allPTurnCoeff)))
    #saveLoc = 'Z:/shared_lab/rotem/pythonCode/GaussianDataRProjections.mat'
    print(noise_level)
    for i in range(len(allpTurnNegGrads)):
         
         for j in range(len(allPTurnCoeff)):
             thisRes = worm_fight_gaussian_grad(allpTurnNegGrads[i], allPTurnCoeff[j],numberOfWorms,memory_len,noise_level)
             memoryWormStepNum[i,j] = thisRes[0]
             regularWormStepNum[i,j] = thisRes[1]
             memoryWormMeanProjection[i,j] = thisRes[2]
             regularWormMeanProjection[i,j] = thisRes[3]
             empiricTurnProbs[i,j] = thisRes[4]
             print((noise_level,i,j))

    resDict = {}
    resDict['memoryWormStepNum'] = memoryWormStepNum
    resDict['regularWormStepNum'] = regularWormStepNum
    resDict['memoryWormMeanProjection'] = memoryWormMeanProjection
    resDict['regularWormMeanProjection'] = regularWormMeanProjection
    resDict['empiricPosTurnRate'] = empiricTurnProbs
    resDict['wormNum'] = numberOfWorms
    resDict['allpTurnNegGrads'] = allpTurnNegGrads
    resDict['allPTurnCoeff'] = allPTurnCoeff
    resDict['memoryLen'] = memory_len
    resDict['numberOfWorms'] = numberOfWorms
    resDict['noiseLevel'] = noise_level


    scipy.io.savemat(saveLoc+"\\simulation_results_noise_"+str(noise_level).replace('.','_'), resDict)

    return [memoryWormStepNum, regularWormStepNum, memoryWormMeanProjection, regularWormMeanProjection, empiricTurnProbs]


def tracksMaking():
    saveLoc = 'c:/temp/Tracks.mat'
    resDict = {}
    wormNum = 5
    noise = 0.001
    myGrad = GaussianGradient(100,noise)

    boundsR = 30
    wormGrad = GaussianGradient(100,noise)
    startX = 300
    startLoc = Loc(startX, 0)



    gradDist = 300
    myLoc = Loc(-1*gradDist, 0)
    memXs =[]
    memYs = []
    regXs =[]
    regYs = []
    pTurnPos =[]
    negPTurn = 0.8
    stepNum = 1500

    for counter in xrange(0, wormNum):
        thisStartOrientation = (random.random() - 0.5) * 2 * math.pi
        # myGrad = LinearGradient(1, 0)


        myWorm = MemoryBiasedRandomWorm(myGrad, 30,negPTurn , myLoc, 1.1, thisStartOrientation)
        for i in range(1, gradDist):
            myWorm.make_step()
        pPos = myWorm.get_positiveGradTurnRate()
        for i in range(1, stepNum-gradDist):
            myWorm.make_step()
        myWorm.plot_track()
        thisMemLocs = myWorm.get_allLocs()


        all_xs = map(lambda x: x.get_x(), thisMemLocs)
        memXs.append(all_xs)
        all_ys = map(lambda x: x.get_y(), thisMemLocs)
        memYs.append(all_ys)


        #resDict['Y'] = all_ys
        #resDict['posTurnRage'] = self.get_positiveGradTurnRate()


        # myWorm.save_track(saveLoc+'mem'+str(counter)+'.mat')
        pTurnPos.append(pPos)
        myRegWorm = BiasedRandomWorm(myGrad, pPos , negPTurn, myLoc, thisStartOrientation)
        for i in range(1, stepNum):
            myRegWorm.make_step()
        myRegWorm.plot_track()
        print (pPos)
        pl.show()
        thisRegLocs = myRegWorm.get_allLocs()
        all_xs = map(lambda x: x.get_x(), thisRegLocs)
        regXs.append(all_xs)
        all_ys = map(lambda x: x.get_y(), thisRegLocs)
        regYs.append(all_ys)

    resDict['memX'] = memXs
    resDict['memY'] = memYs
    resDict['regY'] = regYs
    resDict['regX'] = regXs
    resDict['posTurnRates'] = pTurnPos
    resDict['negativeTurnRate'] = negPTurn
    scipy.io.savemat(saveLoc, resDict)




#saveLoc = 'H:/rotem/Dropbox/Eyal_Rotem_gradients/CODE/2DwormSimulator/results/tracks/'


    #myRegWorm.save_track(saveLoc + 'reg' + str(counter) + '.mat')



#myRegWorm.plot_track(2)

#instancelist = [copy.deepcopy(myWorm) for i in range(29)]

#pl.show()
#def worm_fight_lin_grad(memoryLen,pTurnNegGrad,pTurnCoeff,numberOfWorms,numberOfSteps)




#results = linearGradParamScan()


#res = worm_fight_gaussian_grad(1,0.00005,100,memoryLen=50)
#print(res)
#print((res[1]-res[0])/res[1])
#print(worm_fight_lin_grad(200, 1, 0.1, 100, 2000))
