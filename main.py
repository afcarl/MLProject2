# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 13:13:53 2016

@author: JayaramK
"""
import numpy as np
import csv


'''Reading Letor Data'''
letorfeaturesmat = np.zeros((69623,46))
letorlabelsvec = np.zeros((69623,1))

   
#Open the file and read the LeTor data  into  letorfeaturesmat and  letorlabelsvec
with open('Querylevelnorm.txt', 'r') as infile:
    data = infile.read()  
my_list = data.splitlines()

print len(my_list)


for i in range(0,69623):
   words = my_list[i].split()   
   letorlabelsvec[i] = float(words[0])
   for j in range(2,48):
      letorfeaturesmat[i][j-2] = words[j].split(':')[-1]
      
'''shuffling the entire letor data set ''' 
concatenated_letormat = np.column_stack((letorfeaturesmat, letorlabelsvec))   
np.random.shuffle(concatenated_letormat)   
  
letorfeaturesmat  = concatenated_letormat[:, :46]
letorlabelsvec  = concatenated_letormat[:, [46]]


print ("le11"+str(letorlabelsvec[0][0]))
print("le12"+str(letorlabelsvec[56000][0]))

'''Reading synthetic data'''

synthdatafeaturesmat = np.zeros((20000,10))
synthdatalabelsvec = np.zeros((20000,1))


with open('input.csv', 'rU') as csvfile:
      synthdatareader = list(csv.reader(csvfile,delimiter=','))      
      for i in range(0,20000):
         for j in range(0,10):
             synthdatafeaturesmat[i][j] = synthdatareader[i][j]


        

with open('output.csv', 'rU') as csvfile:
    synthlabeldatareader = list(csv.reader(csvfile)) 
    for i in range(0,20000):
        for j in range(0,1):
            synthdatalabelsvec[i][j] = float(synthlabeldatareader[i][j])
            
'''shuffling the entire synthetic data set '''
concatenated_synthdatamat = np.column_stack((synthdatafeaturesmat, synthdatalabelsvec))   
np.random.shuffle(concatenated_synthdatamat)   
  
synthdatafeaturesmat  = concatenated_synthdatamat[:, :10]
synthdatalabelsvec  = concatenated_synthdatamat[:, [10]]            
     
print ("syn1"+str(synthdatalabelsvec[0][0]))
print("syn2"+str(synthdatalabelsvec[18000][0]))






class DataSplitClass:

    trainingrowsize = 0    
    validationrowsize = 0 
    testingrowsize = 0
    colsize = 0
    trainingfeaturesmat= np.matrix([0,0])  
    traininglabelsvec = np.matrix([0,0])
    
    validationfeaturesmat = np.matrix([0,0])
    validationlabelsvec = np.matrix([0,0])
    
    testingfeaturesmat = np.matrix([0,0])     
    testinglabelsvec = np.matrix([0,0])
    
    
    
    def __init__(self, itrainingrowsize, ivalidationrowsize, itestingrowsize,icolsize):
        self.trainingrowsize = itrainingrowsize
        self.validationrowsize = ivalidationrowsize
        self.testingrowsize = itestingrowsize
        self.colsize = icolsize
        
        self.trainingfeaturesmat= np.zeros((self.trainingrowsize,self.colsize))  
        self.traininglabelsvec = np.zeros((self.trainingrowsize,1))
    
        self.validationfeaturesmat = np.zeros((self.validationrowsize,self.colsize))
        self.validationlabelsvec = np.zeros((self.validationrowsize,1))
    
        self.testingfeaturesmat = np.zeros((self.testingrowsize,self.colsize))
        self.testinglabelsvec = np.zeros((self.testingrowsize,1))
    

    def SplitData(self,featuresmat,labelsvec):
     
        for i in range(0,self.trainingrowsize): 
            for j in range (0,self.colsize):
               self.trainingfeaturesmat[i][j] = featuresmat[i][j] 
            for k in range (0,1):     
                self.traininglabelsvec[i][k] = labelsvec[i][k]    
  
        for i in range(0,self.validationrowsize): 
            for j in range (0,self.colsize):
                self.validationfeaturesmat[i][j] =featuresmat[i+self.trainingrowsize][j] 
            for k in range(0,1): 
                self.validationlabelsvec[i][k] = labelsvec[i+self.trainingrowsize][k]        
     
        for i in range(0,self.testingrowsize): 
             for j in range (0,self.colsize):
                self.testingfeaturesmat[i][j] = featuresmat[i+self.trainingrowsize+self.validationrowsize][j] 
             for k in range (0,1): 
                self.testinglabelsvec[i][k] = labelsvec[i+self.trainingrowsize+self.validationrowsize][k]   
        
        return

def calculateSigmaInverse(featuresmat,colsize,M): 
    tempvec = []        
    for i in range(0,colsize):
        variance = np.var(featuresmat[:,i])/10
        if variance != 0: 
            tempvec.append(variance)
        else :
            tempvec.append(0.001)    
    d = np.diag(tempvec)        
    sigmainv = np.linalg.inv(d)  
    return sigmainv     
    
def CalculateBasisFunctionMatrix(featuresmat,rowsize,colsize,M):
    sigmainv = calculateSigmaInverse(featuresmat,colsize,M)
    '''randomly retrieve  M rows from the training set '''
    mujmat = featuresmat[np.random.choice(featuresmat.shape[0],M, replace=False),:]
    phimat = np.zeros((rowsize,M))

    for i in range(0,rowsize):
        for j in range(0,M):
           ''' (x-muj) and (x-uj)Transpose '''
           if j!=0:
               tempmat= np.subtract(featuresmat[i,:],mujmat[j,:])  
               tempmattranspose = np.transpose(tempmat)  
               tempscalar = tempmattranspose.dot(sigmainv).dot(tempmat)
               phimat[i][j] = np.exp(-0.5*tempscalar)     
           else:
               ''' Value in the first column of the basis matrix = 1'''
               phimat[i][j] = 1            
    return phimat      

def ClosedFormEquation(phimat,labelvec,phimatrowsize,lambda_reg,M):
   IdentityMat = np.eye(M)
   lambdaidentityMat = IdentityMat*lambda_reg
   phimattranspose = np.transpose(phimat)
   phimattranspose_prod_phimat =  np.dot(phimattranspose,phimat)
   tempsum = np.add(lambdaidentityMat,phimattranspose_prod_phimat)
   tempvec = np.linalg.inv(tempsum) 
   tempvec_prod_phimattranspose =np.dot(tempvec,phimattranspose)  
   weightvec = np.dot(tempvec_prod_phimattranspose,labelvec) 
   return weightvec

def CalculateErrorWithReg(rowsize,M,phimat,weightvec,labelvec,lambda_reg): 
   weightvec_transpose_mat = np.transpose(weightvec)
   error = 0
   for i in range(0,rowsize):
       squareofdiff = np.square(np.subtract(labelvec[i],np.dot(weightvec_transpose_mat,phimat[i,:])))
       error = error + (squareofdiff) 
   error = error/2
   weightvec_transpose_mat_prod_weightvec = np.dot(weightvec_transpose_mat,weightvec)  
   error =  error +lambda_reg*(weightvec_transpose_mat_prod_weightvec)/2   
   return error
   
def CalculateRMSError(error,rowsize):    
   erms = np.round(np.sqrt((2*error)/rowsize),5)  
   return erms 
   
   

def ClosedFormSoultion(featuresmat,labelvec,rowsize,colsize,M,lambda_reg):
    phimat = CalculateBasisFunctionMatrix(featuresmat,rowsize,colsize,M)
    weightvec = ClosedFormEquation(phimat,labelvec,len(phimat),lambda_reg,M)
    error = CalculateErrorWithReg(rowsize,M,phimat,weightvec,labelvec,lambda_reg)    
    rmserror = CalculateRMSError(error,rowsize)
    return weightvec,rmserror  
   

def RMSError(featuresmat,labelvec,rowsize,colsize,M,lambda_reg,weightvec):
    phimat = CalculateBasisFunctionMatrix(featuresmat,rowsize,colsize,M)
    error = CalculateErrorWithReg(rowsize,M,phimat,weightvec,labelvec,lambda_reg)    
    rmserror = CalculateRMSError(error,rowsize)
    return rmserror  
    
  
        
def StochasticGradientEquation(weightMat,phimat,labelvec,rowsize,learningrate,lambda_reg,M):
    weightvec_transpose_mat = np.transpose(weightMat)
    newweight = np.zeros((M,1))
    deltaW = np.zeros((M,1))
    weightvec_transpose_mat_prod_phimat= np.empty([rowsize])
    currentErms = 0
    ''' Larger values of prev Erms and Current Rms'''
    prevErms = 100
    minerms = 100
    count = 0 
    newweight = weightMat
    
    minweight=  np.zeros((M,1))
    
    for i in range(0,rowsize):
        
        error = CalculateErrorWithReg(rowsize,M,phimat,newweight,labelvec,lambda_reg)
        currentErms = CalculateRMSError(error,rowsize)

        count = count +1         
        
        if count > 60:
            break
        
        if prevErms-currentErms < 0.001:
            if currentErms < minerms:
                minerms = currentErms
                minweight = newweight
                print("Convergence is achieved")
                break
        if currentErms > prevErms:
            learningrate = learningrate/2
            continue     
        prevErms = currentErms        
        
        weightvec_transpose_mat_prod_phimat[i] = np.dot(weightvec_transpose_mat,phimat[i,:])
        diff = labelvec[i]-weightvec_transpose_mat_prod_phimat[i]
        deltaED = diff*phimat[i,:]
        deltaED = -1*deltaED
        lambda_reg_prod_deltaEW = lambda_reg*newweight 
        deltaE = np.zeros((M,1))
        deltaE = deltaED + lambda_reg_prod_deltaEW
            
        deltaW = -1*learningrate*deltaE
           
        newweight = newweight+deltaW     
        
    return  minweight,minerms        
        
        
def StochasticGradientSolution(featuresmat,labelvec,rowsize,colsize,inputRandomWeightMat,learningrate,M,lambda_reg):        
    phimat = CalculateBasisFunctionMatrix(featuresmat,rowsize,colsize,M) 
    weightvec,rmserror = StochasticGradientEquation(inputRandomWeightMat,phimat,labelvec,rowsize,learningrate,lambda_reg,M)    
    return weightvec,rmserror
        
        
letorobj = DataSplitClass(55698,6962,6963,46)  
letorobj.SplitData(letorfeaturesmat,letorlabelsvec)


synthobj = DataSplitClass(16000,2000,2000,10)   
synthobj.SplitData(synthdatafeaturesmat,synthdatalabelsvec)


#CLOSED FORM SOLUZTION
#For Letor Data
  

#Lambda parameters array

lambda_reg =np.array([0.1,0.3,0.5,0.9])  
M  =  np.array([5,7,9,11,13,15])

minrmserror = 100 
minlambda = 2
minM = 100
minweight  = np.empty([])

for i in range(0,6): 
    for j in range (0,4):
        letorweightvec,letor_trainingrmserror = ClosedFormSoultion(letorobj.trainingfeaturesmat,letorobj.traininglabelsvec,letorobj.trainingrowsize,letorobj.colsize,M[i],lambda_reg[j])
        print("M="+str(M[i])) 
        print("lambda_reg="+str(lambda_reg[j])) 
        print("letor_trainingrmserror="+str(letor_trainingrmserror))
        
        #Tuning the hyperparameter in the Validation Set
        letor_validationrmserror = RMSError(letorobj.validationfeaturesmat,letorobj.validationlabelsvec,letorobj.validationrowsize,letorobj.colsize,M[i],lambda_reg[j],letorweightvec)
        print ("letor_validationrmserror="+str(letor_validationrmserror))             
        if letor_validationrmserror<minrmserror:
          minrmserror = letor_validationrmserror
          minlambda =  lambda_reg[j]
          minM = M[i]
          minwweight = letorweightvec
          print("minwweight=") 
          print(minwweight) 
   
print ("Final_M="+str(minM))
print ("Final_lambda="+str(minlambda))  
print ("Validation_min_rmserror="+str(minrmserror))  
print ("minweight=")
print(minweight)

letor_testingrmserror = RMSError(letorobj.testingfeaturesmat,letorobj.testinglabelsvec,letorobj.testingrowsize,letorobj.colsize,minM,minlambda,minweight)
print ("letor_testingrmserror="+str(letor_testingrmserror))              

                  

#For Synthetic Data

M = np.array([3,5,8])  
lambda_reg =np.array([0.1,0.3,0.5,0.9])  
minrmserror = 100 
minlambda = 2
minM = 100
minweight  = np.empty([])

for i in range(0,3): 
    for j in range (0,4):
        synthweightvec,synth_trainingrmserror = ClosedFormSoultion(synthobj.trainingfeaturesmat,synthobj.traininglabelsvec,synthobj.trainingrowsize,synthobj.colsize,M[i],lambda_reg[j])
        print("M="+str(M[i])) 
        print("lambda_reg="+str(lambda_reg[j])) 
        print ("synth_trainingrmserror="+str(synth_trainingrmserror))
        
        #Tuning the hyperparameter in the Validation Set
        synth_validationrmserror = RMSError(synthobj.validationfeaturesmat,synthobj.validationlabelsvec,synthobj.validationrowsize,synthobj.colsize,M[i],lambda_reg[j],synthweightvec)
        print ("synth_validationrmserror="+str(synth_validationrmserror))             
        if synth_validationrmserror<minrmserror:
          minrmserror = synth_validationrmserror
          minlambda =  lambda_reg[j]
          minM = M[i]
          minwweight = synthweightvec
          print ("minweight=")
          print(minwweight)  

print ("min_rmserror="+str(minrmserror))  
print ("min_lambda="+str(minlambda))  
print ("min_M="+str(minM))
print ("minweight=")
print(minweight)  

synth_testingrmserror = RMSError(synthobj.testingfeaturesmat,synthobj.testinglabelsvec,synthobj.testingrowsize,synthobj.colsize,minM,minlambda,minweight)
print ("synth_testingrmserror="+str(synth_testingrmserror))           
                  










#STOCHASTIC GRADIENT SOLUTION


#For Letor data

learningrate_eta = 1
lambda_reg =np.array([0.1,0.3,0.5,0.9])  
M  =  np.array([5,7,9,11,13,15])  
minrmserror = 100 
minlambda = 2
minM = 100
minweight  = np.empty([])

for i in range(0,6): 
    for j in range (0,4):    
        randomweightvec = np.random.random(M[i])   
        letorweightvec,letor_trainingrmserror = StochasticGradientSolution(letorobj.trainingfeaturesmat,letorobj.traininglabelsvec,letorobj.trainingrowsize,letorobj.colsize,randomweightvec,learningrate_eta,M[i],lambda_reg[j])
        print("M="+str(M[i])) 
        print("lambda_reg="+str(lambda_reg[j])) 
        print("letor_trainingrmserror="+str(letor_trainingrmserror))
        
        #Tuning the hyperparameter in the Validation Set
        letor_validationrmserror = RMSError(letorobj.validationfeaturesmat,letorobj.validationlabelsvec,letorobj.validationrowsize,letorobj.colsize,M[i],lambda_reg[j],letorweightvec)
        print ("letor_validationrmserror="+str(letor_validationrmserror))   

                    
        if letor_validationrmserror<minrmserror:
          minrmserror = letor_validationrmserror
          minlambda =  lambda_reg[j]
          minM = M[i]
          minwweight = letorweightvec
          print("minwweight=") 
          print(minwweight) 

print ("min_rmserror="+str(minrmserror))  
print ("min_lambda="+str(minlambda))  
print ("min_M="+str(minM))
print ("minweight=")
print(minweight)  

letor_testingrmserror = RMSError(letorobj.testingfeaturesmat,letorobj.testinglabelsvec,letorobj.testingrowsize,letorobj.colsize,minM,minlambda,minweight)
print ("letor_testingrmserror="+str(letor_testingrmserror))              


#For Synthetic data
learningrate_eta = 1
lambda_reg =np.array([0.1,0.3,0.5,0.9]) 
M  = np.array([3,5,8]) 
minrmserror = 100 
minlambda = 2
minM = 100
minweight  = np.empty([])

for i in range(0,3): 
    for j in range (0,4):
        randomweightvec = np.random.random(M[i])   
        synthweightvec,synth_trainingrmserror = StochasticGradientSolution(synthobj.trainingfeaturesmat,synthobj.traininglabelsvec,synthobj.trainingrowsize,synthobj.colsize,randomweightvec,learningrate_eta,M[i],lambda_reg[j])
        
        print("M="+str(M[i])) 
        print("lambda_reg="+str(lambda_reg[j])) 
        print("synth_trainingrmserror="+str(synth_trainingrmserror))        
        
        #Tuning the hyperparameter in the Validation Set
        synth_validationrmserror = RMSError(synthobj.validationfeaturesmat,synthobj.validationlabelsvec,synthobj.validationrowsize,synthobj.colsize,M[i],lambda_reg[j],synthweightvec)
        print ("synth_validationrmserror="+str(synth_validationrmserror))             
        if synth_validationrmserror<minrmserror:
          minrmserror = synth_validationrmserror
          minlambda =  lambda_reg[j]
          minM = M[i]
          minwweight = synthweightvec
          print ("minweight=")
          print(minwweight)  

print ("min_rmserror="+str(minrmserror))  
print ("min_lambda="+str(minlambda))  
print ("min_M="+str(minM))
print ("minweight=")
print(minweight)  

synth_testingrmserror = RMSError(synthobj.testingfeaturesmat,synthobj.testinglabelsvec,synthobj.testingrowsize,synthobj.colsize,minM,minlambda,minweight)
print ("synth_testingrmserror="+str(synth_testingrmserror))
                  

    