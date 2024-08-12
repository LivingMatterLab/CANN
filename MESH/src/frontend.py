from src.plotting import *
from src.models import *
from src.cont_mech import *
from src.util_functions import *
from matplotlib import gridspec
import json
import keras.backend
import csv
import pandas as pd
def train_canns(dfs, Psi_model_type, alphas, p_in, Region, modelFit_mode, numIter, cann_name, epochs=10000, batch_size=100):

    path2saveResults_canns = os.path.join(get_path(Region, modelFit_mode), cann_name)
    makeDIR(path2saveResults_canns)

    input_data = {"alphas": alphas, "p": p_in, "numIter": numIter, "epochs": epochs, "batch_size": batch_size}
    with open(f'{path2saveResults_canns}/input_parameters.csv', 'w') as handle:
        writer = csv.writer(handle)
        for key, value in input_data.items():
            writer.writerow([key, value])

    P_ut_all, lam_ut_all, P_ut, lam_ut, P_ss, gamma_ss, midpoint = getStressStrain(dfs, Region)

    final_weights = []
    for num in range(numIter):
        # Create new directory
        path2saveResults = os.path.join(path2saveResults_canns, str(num))
        makeDIR(path2saveResults)
        path2saveResults_check = os.path.join(path2saveResults, 'Checkpoints')
        makeDIR(path2saveResults_check)

        # Save weight, r squared, and loss history for entire training process
        full_weight_hist = []
        full_loss_hist = []
        r2s_ten = []
        r2s_com = []
        r2s_ss = []

        # Iterate through regularization weights
        all_metrics = []
        for id_alpha in range(len(alphas)):
            alpha = alphas[id_alpha]  # get regularization weight alpha
            p = 1 if id_alpha == 0 else p_in  # p = 1 for first iteration, then 0.5

            # Use ogden model with exponents and gains as weights
            Psi_model, terms = Psi_model_type(lam_ut_all, gamma_ss, P_ut_all, P_ss, modelFit_mode, alpha, True, p)

            # Psi_model, terms = StrainEnergyCANN(alpha=alpha, p=p, should_normalize=True)
            if id_alpha > 0:  # If second time through, load best weights from previous iteration
                Psi_model.set_weights(full_weight_hist[-1][-1])

            # Create complete model architecture
            model_UT, model_SS, Psi_model, model = modelArchitecture(Region, Psi_model)

            # Load training data
            model_given, input_train, output_train, sample_weights = traindata(modelFit_mode, model_UT, lam_ut, P_ut,
                                                                               model_SS, gamma_ss, P_ss, model, midpoint)
            # model_given.summary(print_fn=print)
            Save_path = path2saveResults + '/model.h5'
            Save_weights = path2saveResults + '/weights'
            path_checkpoint = path2saveResults_check + '/best_weights'

            # Train model
            model_given, history, weight_hist_arr = Compile_and_fit(model_given, input_train, output_train, epochs,
                                                                    path_checkpoint,
                                                                    sample_weights, batch_size)

            model_given.load_weights(path_checkpoint, by_name=False, skip_mismatch=False)
            tf.keras.models.save_model(Psi_model, Save_path, overwrite=True)
            Psi_model.save_weights(Save_weights, overwrite=True)

            # Get and save loss history
            loss_history = history.history['loss']
            full_loss_hist.append(loss_history)  # Append loss history to full_loss_hist

            # Add final weights to model history
            threshold = 1e-3
            model_weights_0 = Psi_model.get_weights()
            model_weights_0 = [model_weights_0[i] if i%2 == 0 or model_weights_0[i] > threshold ** p else 0.0 * model_weights_0[i] for i in range(len(model_weights_0))]
            weight_hist_arr.append(model_weights_0)
            Psi_model.set_weights(model_weights_0)

            Stress_predict_UT = model_UT.predict(lam_ut_all)
            Stress_predict_SS = model_SS.predict(gamma_ss) if len(gamma_ss) > 0 else []


            if Region.startswith('mesh'):
                P_ut_reshape = reshape_input_output_mesh(P_ut_all)
                Stress_predict_reshape = reshape_input_output_mesh(Stress_predict_UT)

                R2s_ut = np.array([[[r2_score_nonnegative(x[0], x[1]) for x in zip(y[0], y[1])] for y in zip(z[0], z[1])]
                          for z in zip(P_ut_reshape, Stress_predict_reshape)])
                # R2s_ut = [r2_score_nonnegative(P_ut_all[i][j], Stress_predict_UT[i][j])
                #           for j in range(len(P_ut_all[i])) for i in range(len(P_ut_all))]
                R2s_ss = [r2_score_nonnegative(P_ss[i][j], Stress_predict_SS[i][j])
                          for j in range(len(P_ss[i])) for i in range(len(P_ss))] if len(P_ss) > 0 else []
                r2s_ten.append(R2s_ut)
                r2s_ss.append(R2s_ss)

                if modelFit_mode.isnumeric():
                    train_modes = [int(x) for x in modelFit_mode]
                    test_modes = [i for i in range(10) if i not in train_modes]

                    n_samples_per_batch = np.sum(np.array([x.shape[0] for z in Stress_predict_reshape for y in z for x in y])) / 10
                    n_samples_train = len(train_modes) * n_samples_per_batch
                    n_samples_test = len(test_modes) * n_samples_per_batch

                    all_losses = np.array(
                        [[[tf.reduce_sum((x[0] - x[1]) ** 2) for x in zip(y[0], y[1])] for y in zip(z[0], z[1])]
                         for z in zip(P_ut_reshape, Stress_predict_reshape)])
                    all_losses = np.sum(all_losses, axis=-1).flatten()
                    train_loss = np.sum(all_losses[train_modes])
                    test_loss = np.sum(all_losses[test_modes])

                    m_params = model.count_params()
                    chi_squared = train_loss / (n_samples_train - m_params) if n_samples_train > m_params else -1
                    metrics = [m_params, train_loss / n_samples_train,
                               test_loss / n_samples_test if n_samples_test > 0 else -1,
                               chi_squared]
                    all_metrics.append(metrics)

                else:
                    n_comp = 0
                    comp_loss = 0

                    tension_losses = np.array([[[tf.reduce_sum((x[0] - x[1]) ** 2) for x in zip(y[0], y[1])] for y in zip(z[0], z[1])]
                              for z in zip(P_ut_reshape, Stress_predict_reshape)])
                    tension_loss = np.sum(tension_losses)
                    shear_losses = np.array([tf.reduce_sum((P_ss[i][j] - Stress_predict_SS[i][j]) ** 2)
                                             for j in range(len(Stress_predict_SS[i])) for i in
                                             range(len(Stress_predict_SS))]) if len(Stress_predict_SS) > 0 else np.array([])
                    shear_loss = np.sum(shear_losses)

                    n_ten = np.sum(np.array([x.shape[0] for z in Stress_predict_reshape for y in z for x in y]))
                    n_shear = np.sum(np.array([Stress_predict_SS[i][j].shape[0]
                                               for j in range(len(Stress_predict_SS[i])) for i in
                                               range(len(Stress_predict_SS))])) if len(Stress_predict_SS) > 0 else 0
                    all_metrics.append(
                        get_metrics(modelFit_mode, model, tension_loss, n_ten, comp_loss, n_comp, shear_loss, n_shear))
            else:
                # Get r squared values for each loading mode based on model predictions
                r2s_ten.append(r2_score_nonnegative(P_ut_all[midpoint:], Stress_predict_UT[midpoint:]))
                r2s_com.append(r2_score_nonnegative(P_ut_all[:(midpoint + 1)], Stress_predict_UT[:(midpoint + 1)]))
                r2s_ss.append(r2_score_nonnegative(P_ss, Stress_predict_SS))
                comp_loss = tf.reduce_sum(
                    (P_ut[:(midpoint + 1)].squeeze() - Stress_predict_UT[:(midpoint + 1)].squeeze()) ** 2)

                tension_loss = tf.reduce_sum((P_ut[midpoint:].squeeze() - Stress_predict_UT[midpoint:].squeeze()) ** 2)
                shear_loss = tf.reduce_sum((P_ss[midpoint:].squeeze() - Stress_predict_SS[midpoint:].squeeze()) ** 2)
                n_shear = midpoint + 1
                n_ten = midpoint + 1
                n_comp = midpoint + 1

                all_metrics.append(
                    get_metrics(modelFit_mode, model, tension_loss, n_ten, comp_loss, n_comp, shear_loss, n_shear))

            # Format of weight list is alternating exponent, gain, exponent, gain, ...
            if id_alpha == 0 and p_in < 1:
                # If this is the first time through training, we need to convert the weights from the L1 version of
                # the model to the L0.5 version. Because the L0.5 version squares each gain, we need to take the
                # square root of every other element starting with second
                full_weight_hist = [[[x[i] ** (1 - (1-p_in) * (i % 2)) for i in range(len(x))] for x in weight_hist_arr]]
            else:
                full_weight_hist.append(weight_hist_arr)

        # Save final optimized weights for uniqueness plot
        final_weights.append(full_weight_hist[-1][-1])

        # Save metrics
        all_metrics = np.array(all_metrics)
        all_metrics = pd.DataFrame(all_metrics)
        all_metrics.to_csv(os.path.join(path2saveResults, "all_metrics.csv"))

        # Save data for animations
        input_data = {"weight_hist": full_weight_hist, "loss_hist": full_loss_hist, "terms": terms, "Lalphas": alphas,
                      "Region": Region, "r2": [r2s_ten, r2s_com, r2s_ss]}
        with open(f'{path2saveResults}/training.pickle', 'wb') as handle:
            pickle.dump(input_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sorted_weights = np.squeeze(np.array([sort_weights(weights) for weights in final_weights]))
    np.save(f'{path2saveResults_canns}/box_whisker.npy', sorted_weights)

    tf.keras.backend.clear_session()





