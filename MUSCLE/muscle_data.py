import pandas as pd
import numpy as np


def load_data():
    inputPath1 = r'Data/muscle_mag_vals.xls'
    inputPath2 = r'Data/muscle_rate_vals.xls'

    inputData = np.zeros((5, 50, 2))
    outputData = np.zeros((5, 50, 1))

    df = pd.read_excel(inputPath1, sheet_name=0)
    inputData[0, :, :] = df[['t-s', 'str']].to_numpy()
    outputData[0, :, 0] = df['Cauchy-kPa'].to_numpy()

    df1 = pd.read_excel(inputPath1, sheet_name=1)
    inputData[1, :, :] = df1[['t', 'str']].to_numpy()
    outputData[1, :, 0] = df1['y'].to_numpy()

    df2 = pd.read_excel(inputPath1, sheet_name=2)
    inputData[2, :, :] = df2[['x', 'str']].to_numpy()
    outputData[2, :, 0] = df2['y'].to_numpy()

    df3 = pd.read_excel(inputPath2, sheet_name=0)
    inputData[3, :, :] = df3[['t-s', 'str']].to_numpy()
    outputData[3, :, 0] = df3['Cauchy-kPa'].to_numpy()

    df4 = pd.read_excel(inputPath2, sheet_name=2)
    inputData[4, :, :] = df4[['x', 'str']].to_numpy()
    outputData[4, :, 0] = df4['y'].to_numpy()

    inputTrain = np.zeros((4, 49, 2))
    outputTrain = np.zeros((4, 49, 1))
    inputTest = np.zeros((1, 49, 2))
    outputTest = np.zeros((1, 49, 1))

    for i in range(inputTrain.shape[0]):
        for j in range(inputTrain.shape[1]):
            inputTrain[i, j, 0] = inputData[i, j + 1, 0] - inputData[i, j, 0]
            inputTrain[i, j, 1] = inputData[i, j + 1, 1]
            outputTrain[i, j, 0] = outputData[i, j + 1, 0]

    for i in range(inputTest.shape[0]):
        for j in range(inputTest.shape[1]):
            index = i + inputTrain.shape[0]
            inputTest[i, j, 0] = inputData[index, j + 1, 0] - inputData[index, j, 0]
            inputTest[i, j, 1] = inputData[index, j + 1, 1]
            outputTest[i, j, 0] = outputData[index, j + 1, 0]

    inputAll = np.concatenate((inputTrain, inputTest))
    outputAll = np.concatenate((outputTrain, outputTest))

    trainingWeights = np.zeros(inputTrain.shape[0])

    total = 0
    for i in range(inputTrain.shape[0]):
        meanVal = np.mean(outputTrain[i])
        trainingWeights[i] = 1 / np.abs(meanVal)
        total = total + (1 / np.abs(meanVal))

    trainingWeights = trainingWeights / total

    return inputTrain, outputTrain, inputTest, outputTest, inputAll, outputAll, trainingWeights


def load_resampled_data():
    inputPath1 = r'Data/muscle_mag_vals.xls'
    inputPath2 = r'Data/muscle_rate_vals.xls'

    inputData = np.zeros((5, 18, 2))
    outputData = np.zeros((5, 18, 1))

    df = pd.read_excel(inputPath1, sheet_name=0)
    inputData[0, :, :] = df[['t_r', 'str_r']].to_numpy()[0:18, :]
    outputData[0, :, 0] = df['sig_r'].to_numpy()[0:18]

    df1 = pd.read_excel(inputPath1, sheet_name=1)
    inputData[1, :, :] = df1[['t_r', 'str_r']].to_numpy()[0:18, :]
    outputData[1, :, 0] = df1['sig_r'].to_numpy()[0:18]

    df2 = pd.read_excel(inputPath1, sheet_name=2)
    inputData[2, :, :] = df2[['t_r', 'str_r']].to_numpy()[0:18, :]
    outputData[2, :, 0] = df2['sig_r'].to_numpy()[0:18]

    df3 = pd.read_excel(inputPath2, sheet_name=0)
    inputData[3, :, :] = df3[['t_r', 'str_r']].to_numpy()[0:18, :]
    outputData[3, :, 0] = df3['sig_r'].to_numpy()[0:18]

    df4 = pd.read_excel(inputPath2, sheet_name=2)
    inputData[4, :, :] = df4[['t_r', 'str_r']].to_numpy()[0:18, :]
    outputData[4, :, 0] = df4['sig_r'].to_numpy()[0:18]

    inputTrain = np.zeros((4, 17, 2))
    outputTrain = np.zeros((4, 17, 1))
    inputTest = np.zeros((1, 17, 2))
    outputTest = np.zeros((1, 17, 1))

    for i in range(inputTrain.shape[0]):
        for j in range(inputTrain.shape[1]):
            inputTrain[i, j, 0] = inputData[i, j + 1, 0] - inputData[i, j, 0]
            inputTrain[i, j, 1] = inputData[i, j + 1, 1]
            outputTrain[i, j, 0] = outputData[i, j + 1, 0]

    for i in range(inputTest.shape[0]):
        for j in range(inputTest.shape[1]):
            index = i + inputTrain.shape[0]
            inputTest[i, j, 0] = inputData[index, j + 1, 0] - inputData[index, j, 0]
            inputTest[i, j, 1] = inputData[index, j + 1, 1]
            outputTest[i, j, 0] = outputData[index, j + 1, 0]

    inputAll = np.concatenate((inputTrain, inputTest))
    outputAll = np.concatenate((outputTrain, outputTest))
    trainingWeights = np.zeros(inputTrain.shape[0])

    total = 0
    for i in range(inputTrain.shape[0]):
        meanVal = np.mean(outputTrain[i])
        trainingWeights[i] = 1 / np.abs(meanVal)
        total = total + (1 / np.abs(meanVal))

    trainingWeights = trainingWeights / total

    return inputTrain, outputTrain, inputTest, outputTest, inputAll, outputAll, trainingWeights


def load_axon_data():
    inputPath = r'Data/axonData.xls'

    inputData = np.zeros((2, 27, 2))
    outputData = np.zeros((2, 27, 1))

    df = pd.read_excel(inputPath, sheet_name=1)
    inputData[0, :, :] = df[['time_s', 'stretch']].to_numpy()[0:27, :]
    outputData[0, :, 0] = df['stress_kpa'].to_numpy()[0:27]

    df1 = pd.read_excel(inputPath, sheet_name=0)
    inputData[1, :, :] = df1[['time_s', 'str']].to_numpy()[0:27, :]
    outputData[1, :, 0] = df1['stress_kpa'].to_numpy()[0:27]


    inputTrain = np.zeros((1, 26, 2))
    outputTrain = np.zeros((1, 26, 1))
    inputTest = np.zeros((1, 26, 2))
    outputTest = np.zeros((1, 26, 1))

    for i in range(inputTrain.shape[0]):
        for j in range(inputTrain.shape[1]):
            inputTrain[i, j, 0] = inputData[i, j + 1, 0] - inputData[i, j, 0]
            inputTrain[i, j, 1] = inputData[i, j + 1, 1]
            outputTrain[i, j, 0] = outputData[i, j + 1, 0]

    for i in range(inputTest.shape[0]):
        for j in range(inputTest.shape[1]):
            index = i + inputTrain.shape[0]
            inputTest[i, j, 0] = inputData[index, j + 1, 0] - inputData[index, j, 0]
            inputTest[i, j, 1] = inputData[index, j + 1, 1]
            outputTest[i, j, 0] = outputData[index, j + 1, 0]

    inputAll = np.concatenate((inputTrain, inputTest))
    outputAll = np.concatenate((outputTrain, outputTest))
    trainingWeights = np.zeros(inputTrain.shape[0])

    total = 0
    for i in range(inputTrain.shape[0]):
        meanVal = np.mean(outputTrain[i])
        trainingWeights[i] = 1 / np.abs(meanVal)
        total = total + (1 / np.abs(meanVal))

    trainingWeights = trainingWeights / total

    return inputTrain, outputTrain, inputTest, outputTest, inputAll, outputAll, trainingWeights


def load_smooth_axon_data():
    inputPath = r'Data/axonData.xls'

    inputData = np.zeros((2, 29, 2))
    outputData = np.zeros((2, 29, 1))

    df = pd.read_excel(inputPath, sheet_name=1)
    inputData[0, :, :] = df[['time_sm', 'stretch_sm']].to_numpy()[0:29, :]
    outputData[0, :, 0] = df['stress_sm'].to_numpy()[0:29]

    df1 = pd.read_excel(inputPath, sheet_name=0)
    inputData[1, :, :] = df1[['time_s', 'str']].to_numpy()[0:29, :]
    outputData[1, :, 0] = df1['stress_kpa'].to_numpy()[0:29]

    inputTrain = np.zeros((1, 28, 2))
    outputTrain = np.zeros((1, 28, 1))
    inputTest = np.zeros((1, 28, 2))
    outputTest = np.zeros((1, 28, 1))

    for i in range(inputTrain.shape[0]):
        for j in range(inputTrain.shape[1]):
            inputTrain[i, j, 0] = inputData[i, j + 1, 0] - inputData[i, j, 0]
            inputTrain[i, j, 1] = inputData[i, j + 1, 1]
            outputTrain[i, j, 0] = outputData[i, j + 1, 0]

    for i in range(inputTest.shape[0]):
        for j in range(inputTest.shape[1]):
            index = i + inputTrain.shape[0]
            inputTest[i, j, 0] = inputData[index, j + 1, 0] - inputData[index, j, 0]
            inputTest[i, j, 1] = inputData[index, j + 1, 1]
            outputTest[i, j, 0] = outputData[index, j + 1, 0]

    inputAll = np.concatenate((inputTrain, inputTest))
    outputAll = np.concatenate((outputTrain, outputTest))
    trainingWeights = np.zeros(inputTrain.shape[0])

    total = 0
    for i in range(inputTrain.shape[0]):
        meanVal = np.mean(outputTrain[i])
        trainingWeights[i] = 1 / np.abs(meanVal)
        total = total + (1 / np.abs(meanVal))

    trainingWeights = trainingWeights / total

    return inputTrain, outputTrain, inputTest, outputTest, inputAll, outputAll, trainingWeights

def load_PC12_axon_data():
    inputPath = r'Data/axonData.xls'

    inputData = np.zeros((1, 31, 2))
    outputData = np.zeros((1, 31, 1))

    df = pd.read_excel(inputPath, sheet_name=2)
    inputData[0, :, :] = df[['time_s', 'stretch']].to_numpy()[0:31, :]
    outputData[0, :, 0] = df['stress_kpa'].to_numpy()[0:31]


    inputTrain = np.zeros((1, 30, 2))
    outputTrain = np.zeros((1, 30, 1))

    for i in range(inputTrain.shape[0]):
        for j in range(inputTrain.shape[1]):
            inputTrain[i, j, 0] = inputData[i, j + 1, 0] - inputData[i, j, 0]
            inputTrain[i, j, 1] = inputData[i, j + 1, 1]
            outputTrain[i, j, 0] = outputData[i, j + 1, 0]

    return inputTrain, outputTrain