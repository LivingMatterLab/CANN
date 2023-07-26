from plotting_tools import *
from viscRNN import *
from muscle_data import *
from tensorflow.python.framework.ops import disable_eager_execution

if __name__ == '__main__':
    savePath = 'Results/'
    trialName = 'muscle1'
    numHistoryVars = 10  # number of Prony terms
    num_pStr_Units = 10  # number of principal stretch terms in the initial stored energy function
    L2 = 0.0000001  # regularization strength for the initial stored energy function
    r_prony = 0.00  # regularization strength for the relaxation function

    # load muscle data
    inputTrain, outputTrain, inputTest, outputTest, inputAll, outputAll, trainingWeights = load_resampled_data()

    # example of training invariant-based model
    model = build_inv(numHistoryVars, L2, r_prony)

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    history = model.fit([np.expand_dims(inputTrain[:, :, 1], axis=2), np.expand_dims(inputTrain[:, :, 0], axis=2)],
                        outputTrain, batch_size=1, validation_data=([np.expand_dims(inputTest[:, :, 1], axis=2),
                                                                     np.expand_dims(inputTest[:, :, 0], axis=2)],
                                                                    outputTest),
                        sample_weight=trainingWeights, epochs=50)

    model.save_weights(savePath + trialName + '_inv.tf')

    plotLoss(history, savePath, trialName + '_inv')
    plotPred(model, inputAll, outputAll, savePath, trialName + '_inv')

    # example of training principal stretch-based model
    disable_eager_execution()  # use this command for the principal stretch-based model only

    model = build_pStr(num_pStr_Units, numHistoryVars)

    model.compile(loss=tf.keras.losses.LogCosh(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    history = model.fit([np.expand_dims(inputTrain[:, :, 1], axis=2), np.expand_dims(inputTrain[:, :, 0], axis=2)],
                        outputTrain, batch_size=1, validation_data=([np.expand_dims(inputTest[:, :, 1], axis=2),
                                                                     np.expand_dims(inputTest[:, :, 0], axis=2)],
                                                                    outputTest),
                        sample_weight=trainingWeights, epochs=50)

    model.save_weights(savePath + trialName + '_pStr.tf')

    plotLoss(history, savePath, trialName + '_pStr')
    plotPred(model, inputAll, outputAll, savePath, trialName + '_pStr')
