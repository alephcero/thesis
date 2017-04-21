#libraries
import pandas as pd
import numpy as np
import os
import sys
import pysal as ps
#import simpledbf
import math
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split


def getEPHInd(trimestre, path):
    '''
    This functions takes as parameters
    trimestre: t110 = primer trimestre 2010
    path: where to store data
    '''
    #if data directory doesn't exists:
    if ~(os.path.isdir(path)):
        #create it
        print 'Directory non existent, creating ...'
        os.system('mkdir '+path)

    #if original dbf file doesn't exists:
    if not os.path.isfile(path + '/' + trimestre + '/Individual_'+ trimestre +'.dbf'):

        #download the zip file, 
        print 'File is not there, downloading...'
        os.system("curl -O http://www.indec.gob.ar/ftp/cuadros/menusuperior/eph/" + trimestre + "_dbf.zip")

        #mote it to /data
        print 'Moving file...'
        os.system("mv " + trimestre + "_dbf.zip " + path)

        #unzipit and
        print 'Unziping file...'
        os.system("unzip " + path + '/' + trimestre + "_dbf.zip"  ' -d ' + path + '/' + trimestre)

        #remove zip file
        print 'Removing zip...'
        os.system('rm ' + path + '/' + trimestre + "_dbf.zip")
    else:
        print 'Original files in directory:', path

        
def dbf2DF(dbfile, upper=True): #Reads in DBF files and returns Pandas DF
    '''
    Arguments
    ---------
    dbfile  : DBF file - Input to be imported
    upper   : Condition - If true, make column heads upper case
    '''
    db = ps.open(dbfile) #Pysal to open DBF
    d = {col: db.by_col(col) for col in db.header} #Convert dbf to dictionary
    pandasDF = pd.DataFrame(db[:]) #Convert to Pandas DF
    pandasDF.columns=db.header
    pandasDF.dropna(inplace=True)
    return pandasDF


def readEPHInd(trimestre, path):
    #read file
    dbf = simpledbf.Dbf5(path  + '/' + trimestre + '/Individual_'+ trimestre +'.dbf', codec='latin1')
    indRaw = dbf.to_dataframe()
    return indRaw


def cleanEPHInd(dataset):
    indClean = dataset.loc[dataset.REGION == 1,['CODUSU',
                        'NRO_HOGAR',
                        'COMPONENTE',
                        'AGLOMERADO',
                        'PONDERA',
                        'CH03',
                        'CH04',
                        'CH06',
                        'CH12', ## schoolLevel
                        'CH13',
                        'CH14',
                        'ESTADO',
                        'CAT_OCUP',
                        'CAT_INAC',
                        'ITF',
                        'IPCF',
                        'P47T',
                        'P21']]

    indClean.columns = ['CODUSU',
                        'NRO_HOGAR',
                        'COMPONENTE',
                        'AGLOMERADO',
                        'PONDERA',
                        'familyRelation',
                        'female',
                        'age',
                        'schoolLevel',## schoolLevel
                        'finishedYear',
                        'lastYear',
                        'activity',
                        'empCond',
                        'unempCond',
                        'ITF',
                        'IPCF',
                  'P47T',
                  'P21']
    indClean.index =range(0,indClean.shape[0])
    return indClean
        
def categorizeInd(df):
    df.female = (df.female == 2).astype(int)
    df.schoolLevel.replace(to_replace=[99], value=[np.nan] , inplace=True, axis=None) 
    df.lastYear.replace(to_replace=[98,99], value=[np.nan, np.nan] , inplace=True, axis=None)
    df.activity.replace(to_replace=[0], value=[np.nan] , inplace=True, axis=None)
    df.empCond.replace(to_replace=[0], value=[np.nan] , inplace=True, axis=None)
    df.unempCond.replace(to_replace=[0], value=[np.nan] , inplace=True, axis=None)
    return df



def schoolYears(dataset):
    '''
    This function takes a dataset with
    schoolLevel: last level of school attended
    finishedYear: if the person finished that level
    lastYear: last year of that level aproved
    and returns a new dataset with the amount of school years for each level
    '''
    primary = []
    secondary = []
    university = []
    
    for i in range(dataset.shape[0]):
        
        #kinder, special or no school
        if ((dataset['schoolLevel'][i] > 8) | (dataset['schoolLevel'][i] < 2)):
            primary_i = 0
            secondary_i = 0
            university_i = 0
        
        #from primary to university
        else:
            #finished their level
            if dataset['finishedYear'][i] == 1:
                
                #finish primary
                if ((dataset['schoolLevel'][i] == 2) | (dataset['schoolLevel'][i] == 3)):
                    primary_i = 7
                    secondary_i = 0
                    university_i = 0
                
                #finish seconday
                elif ((dataset['schoolLevel'][i] == 4) | (dataset['schoolLevel'][i] == 5)):
                    primary_i = 7
                    secondary_i = 5
                    university_i = 0
                
                #finish college
                elif dataset['schoolLevel'][i] == 6:
                    primary_i = 7
                    secondary_i = 5
                    university_i = 3
                    
                #finish university
                elif ((dataset['schoolLevel'][i] == 7) | (dataset['schoolLevel'][i] == 8)):
                    primary_i = 7
                    secondary_i = 5
                    university_i = 5
            # didn't finish
            elif dataset['finishedYear'][i] == 2:
                
                #not finish primary
                if dataset['schoolLevel'][i] == 2:
                    if dataset['lastYear'][i] > 90:
                        primary_i = 0
                    elif dataset['lastYear'][i] > 6:
                        primary_i = 6
                    elif pd.isnull(dataset['lastYear'][i]):
                        primary_i = 3
                    else:
                        primary_i = dataset['lastYear'][i]
                    secondary_i = 0
                    
                    university_i = 0
                
                #not finish EGB
                elif dataset['schoolLevel'][i] == 3:
                    if dataset['lastYear'][i] > 90:
                        primary_i = 0
                        secondary_i = 0
                    elif pd.isnull(dataset['lastYear'][i]):
                        primary_i = 3
                        secondary_i = 0
                    elif dataset['lastYear'][i] > 7:
                        primary_i = 7
                        secondary_i = dataset['lastYear'][i] - 7
                    else:
                        primary_i = dataset['lastYear'][i]
                        secondary_i = 0
                        
                    university_i = 0

                    
                #not finish Secondary
                elif dataset['schoolLevel'][i] == 4:
                    if dataset['lastYear'][i] > 90:
                        secondary_i = 0
                    elif pd.isnull(dataset['lastYear'][i]):
                        secondary_i = 2
                    elif dataset['lastYear'][i] > 5:
                        secondary_i = 5
                    else:
                        secondary_i = dataset['lastYear'][i]
                                            
                    primary_i = 7
                    university_i = 0
                
                #not finish polimodal
                elif dataset['schoolLevel'][i] == 5:
                    if dataset['lastYear'][i] > 90:
                        secondary_i = 0
                    elif pd.isnull(dataset['lastYear'][i]):
                        secondary_i = 1
                    elif dataset['lastYear'][i] > 2:
                        secondary_i = 4
                    else:
                        secondary_i = dataset['lastYear'][i]
                    
                    primary_i = 7    
                    university_i = 0
               
                
                #not finish college
                elif dataset['schoolLevel'][i] == 6:
                    if dataset['lastYear'][i] > 90:
                        university_i = 2
                    elif pd.isnull(dataset['lastYear'][i]):
                        university_i = 1
                    elif dataset['lastYear'][i] > 3:
                        university_i = 3
                    else:
                        university_i = dataset['lastYear'][i]
                        
                    primary_i = 7
                    secondary_i = 5
                    
                    
                #no finish university
                elif ((dataset['schoolLevel'][i] == 7) | (dataset['schoolLevel'][i] == 8)):
                    if dataset['lastYear'][i] > 90:
                        university_i = 3
                    elif pd.isnull(dataset['lastYear'][i]):
                        university_i = 2
                    elif dataset['lastYear'][i] > 5:
                        university_i = 5
                    elif math.isnan(dataset['lastYear'][i]):
                        university_i = 2

                    else:
                        university_i = dataset['lastYear'][i]
                        
                    primary_i = 7
                    secondary_i = 5

                #last year proved
                
                
            #don't know
            else:
                primary_i = 0
                secondary_i = 0
                university_i = 0
            
        
        #add values to list
        primary.append(primary_i)
        secondary.append(secondary_i)
        university.append(university_i)
    
    dataset['primary'] = primary
    dataset['secondary'] = secondary
    dataset['university'] = university
    
    return dataset


def make_dummyInd(data):
    data['male_14to24'] = ((data.female == 0) & ((data.age >= 14) & (data.age <= 24))).astype(int) 
    data['male_25to34'] = ((data.female == 0) & ((data.age >= 25 ) & ( data.age <= 34))).astype(int)
    data['female_14to24'] = ((data.female == 1) & ((data.age >= 14 ) & ( data.age <= 24))).astype(int)
    data['female_25to34'] = ((data.female == 1) & ((data.age >= 25 ) & ( data.age <= 34))).astype(int)
    data['female_35more'] = ((data.female == 1) & ((data.age >= 35 ))).astype(int)    
    return data


def createVariablesInd(dataset):
    #dataset.drop_duplicates(inplace=True)
    dataset['education'] = dataset.primary + dataset.secondary + dataset.university
    dataset['education2'] = dataset['education'] ** 2
    dataset['age2'] = dataset['age'] ** 2
    dataset['id'] = (dataset.CODUSU.astype(str) + dataset.NRO_HOGAR.astype(str))
    dataset.P47T.replace(to_replace=[0], value=[1] , inplace=True, axis=None)
    dataset.P21.replace(to_replace=[0], value=[1] , inplace=True, axis=None)
    dataset['lnIncome']= np.log(dataset.P21)
    dataset['lnIncomeT']= np.log(dataset.P47T)
    return dataset

def runModel(dataset, income = 'lnIncome',
              variables = [
        'primary','secondary','university',
        'male_14to24','male_25to34',
        'female_14to24', 'female_25to34', 'female_35more']):
    
    '''
    This function takes a data set, runs a model according to specifications,
    and returns the model, printing the summary
    '''
    #prepare data
    y = dataset[income].copy().values
    X = sm.add_constant(dataset.copy().loc[:,variables].values)
    w = dataset.PONDERA.copy().values
    
    #run model
    lm = sm.WLS(y, X, weights=1. / w).fit()
    print lm.summary()
    for i in range(1,len(variables)+1):
        print 'x%d: %s' % (i,variables[i-1])
    #testing within sample
    '''
    R_IS=[]
    R_OS=[]
    nCross=1000
    
    for i in range(nCross):
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.33)
        lmCross = sm.WLS(y_train, X_train, weights=1. / w_train, hasconst=False).fit()        
        R_IS.append(1-((np.asarray(lmCross.predict(exog = X_train))-y_train)**2).sum()/((y_train-np.mean(y_train))**2).sum())                                                                     
        R_OS.append(1-((np.asarray(lmCross.predict(exog = X_test))-y_test)**2).sum()/((y_test-np.mean(y_test))**2).sum())
    print("IS R-squared for {} times is {}".format(nCross,np.mean(R_IS)))
    print("OS R-squared for {} times is {}".format(nCross,np.mean(R_OS)))
    '''
    return lm

def predictModelo(modelo, primary = 0, secondary = 0, university = 0,
                  male_14to24 = 0, male_25to34 = 0,
                  female_14to24 = 0, female_25to34 = 0, female_35tomore = 0
                  ):
    
    '''
    Esta funcion toma un vector booleano de variables 1 o 0 para las variables dummy
    y devuelve el valor del modelo. 
    La base es hombre mayor de 35 anios con 0 anios de escolaridad
    '''
    observacion = [primary,secondary,university,male_14to24,male_25to34,female_14to24,female_25to34,female_35tomore]
    resultado = np.exp(np.dot(modelo.params[1:],observacion) + modelo.params[0])
    return resultado

def adultoEquivalente(row):
    age = row.age
    female = row.female
    resultado = np.nan
    if (age < 10) : #si es menor a 10 no hay diferencia de genero
        if age < 1 :
            resultado = 0.315 # se pondera entre 0.28 y 0.35 porque EPH 2010 no lo da en meses
        if age == 1 :
            resultado = 0.37
        if age == 2 :
            resultado = 0.46
        if age == 3 :
            resultado = 0.51
        if age == 4 :
            resultado = 0.55
        if age == 5 :
            resultado = 0.6
        if age == 6 :
            resultado = 0.64         
        if age == 7 :
            resultado = 0.66
        if age == 8 :
            resultado = 0.68
        if age == 9 :
            resultado = 0.69
    else: #si es mayor a 10
        if female == 0: #si es varon
            if age == 10 :
                resultado = 0.79
            elif age == 11 :
                resultado = 0.82
            elif age == 12 :
                resultado = 0.85
            elif age == 13 :
                resultado = 0.90
            elif age == 14 :
                resultado = 0.96            
            elif age == 15 :
                resultado = 1.
            elif age == 16 :
                resultado = 1.03
            elif age == 17 :
                resultado = 1.04
            elif ((age >= 18) & (age <= 29)):
                resultado = 1.02
            elif ((age >= 30) & (age <= 45)):
                resultado = 1.0
            elif ((age >= 46) & (age <= 60)):
                resultado = 1.0
            elif ((age >= 61) & (age <= 75)):
                resultado = 0.83
            elif age > 75:
                resultado = 0.74
            else:
                resultado = np.nan
        else: #si es mujer
            if age == 10 :
                resultado = 0.70
            elif age == 11 :
                resultado = 0.72
            elif age == 12 :
                resultado = 0.74
            elif age == 13 :
                resultado = 0.76
            elif age == 14 :
                resultado = 0.76            
            elif age == 15 :
                resultado = 0.77
            elif age == 16 :
                resultado = 0.77
            elif age == 17 :
                resultado = 0.77
            elif ((age >= 18) & (age <= 29)):
                resultado = 0.76
            elif ((age >= 30) & (age <= 45)):
                resultado = 0.77
            elif ((age >= 46) & (age <= 60)):
                resultado = 0.76
            elif ((age >= 61) & (age <= 75)):
                resultado = 0.67
            elif age > 75:
                resultado = 0.63
            else:
                resultado = np.nan
    return resultado
            
            
            
            
            
