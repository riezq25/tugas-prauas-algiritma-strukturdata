# json to dictionary
import json
import numpy as np


def getDataset():
    dataset = []
    with open('data_latih.json') as data_file:
        data = json.load(data_file)
        for i in data:
            dataset.append(i)
    return dataset


def splitClassLabel(dataset):
    data = {}
    for i in dataset:
        if i['kelas'] not in data:
            data[i['kelas']] = []
        data[i['kelas']].append(i)
    return data


def averageClassLabel(dataset):
    averageClass = {}
    for i in dataset:
        temp = 0
        humid = 0

        for j in dataset[i]:
            temp += j['temp']
            humid += j['humid']

        humid /= len(dataset[i])
        temp /= len(dataset[i])

        averageClass[i] = {'humid': humid, 'temp': temp}

    return averageClass


def globalAverage(dataset):
    humid = 0
    temp = 0
    for i in dataset:
        humid += i['humid']
        temp += i['temp']

    return {'humid': humid/len(dataset), 'temp': temp/len(dataset)}


def meanCorrectedClass(splitedDataset, globalAverages):
    meanCorrectedClass = {}
    for i in splitedDataset:
        meanCorrectedClass[i] = []
        for j in splitedDataset[i]:
            meanCorrectedClass[i].append(
                {'humid': j['humid'] - globalAverages['humid'], 'temp': j['temp'] - globalAverages['temp']})
    return meanCorrectedClass

# numpy matrix


def meanCorrectedClassToMatrix(meanCorrected):
    meanCorrectedMatrix = {}
    for i in meanCorrected:
        meanCorrectedMatrix[i] = []
        for j in meanCorrected[i]:
            meanCorrectedMatrix[i].append([j['humid'], j['temp']])
        meanCorrectedMatrix[i] = np.matrix(meanCorrectedMatrix[i])
    return meanCorrectedMatrix


def covarianceMatrix(meanCorrectedMatrix):
    covarianceMatrix = {}
    for i in meanCorrectedMatrix:
        covarianceMatrix[i] = meanCorrectedMatrix[i].transpose(
        ) * meanCorrectedMatrix[i] / len(meanCorrectedMatrix[i])
    return covarianceMatrix


def globalCovarianceMatrix(covarianceMatrix):
    return np.matrix([
        [
            (covarianceMatrix[1][0, 0]*3/87 + covarianceMatrix[2][0, 0]*60/87 +
             covarianceMatrix[3][0, 0]*3/87 + covarianceMatrix[4][0, 0]*21/87),
            (covarianceMatrix[1][0, 1]*3/87 + covarianceMatrix[2][0, 1]*60/87 +
             covarianceMatrix[3][0, 1]*3/87 + covarianceMatrix[4][0, 1]*21/87)
        ], [
            (covarianceMatrix[1][1, 0]*3/87 + covarianceMatrix[2][1, 0]*60/87 +
             covarianceMatrix[3][1, 0]*3/87 + covarianceMatrix[4][1, 0]*21/87),
            (covarianceMatrix[1][1, 1]*3/87 + covarianceMatrix[2][1, 1]*60/87 +
             covarianceMatrix[3][1, 1]*3/87 + covarianceMatrix[4][1, 1]*21/87)
        ]
    ])  # type: ignore


def inverseGlobalCovarianceMatrix(globalCovarianceMatrix):
    return np.matrix(globalCovarianceMatrix).I


def priorProbability(splitedDataset):
    priorProbability = {}
    for i in splitedDataset:
        priorProbability[i] = len(splitedDataset[i])/87
    return priorProbability


def descisionFunction(globalCovarianceMatrix, inverseGlobalCovarianceMatrix, priorProbability, averageClass, x):
    descisionFunction = {}
    for i in averageClass:
        side1 = np.matmul(np.matmul(averageClass[i], inverseGlobalCovarianceMatrix), np.matrix(x).transpose())
        side2 = -0.5 * np.matmul(np.matmul(averageClass[i], inverseGlobalCovarianceMatrix), averageClass[i].transpose())
        side3 = np.log(priorProbability[i])
        
        descisionFunction[i] = side1+side2+side3
    return descisionFunction


dataset = getDataset()
splitedDataset = splitClassLabel(dataset)
averageClass = averageClassLabel(splitedDataset)
globalAverages = globalAverage(dataset)
meanCorrected = meanCorrectedClass(splitedDataset, globalAverages)
meanCorrectedMatrix = meanCorrectedClassToMatrix(meanCorrected)
covarianceMatrixs = covarianceMatrix(meanCorrectedMatrix)
globalCovarianceMatrixs = globalCovarianceMatrix(
    covarianceMatrixs)  # type: ignore
inverseGlobalCovarianceMatrixs = inverseGlobalCovarianceMatrix(
    globalCovarianceMatrixs)  # type: ignore
priorProbabilities = priorProbability(splitedDataset)
descisionFunctions = descisionFunction(
    globalCovarianceMatrixs, inverseGlobalCovarianceMatrixs, priorProbabilities, averageClass, [0.5, 0.5])
print(descisionFunctions)
