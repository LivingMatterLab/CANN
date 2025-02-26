import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm
import pathlib
import matplotlib.pyplot as plt


def Stress_xx(c1, c2, lx, ly):  # planar biaxial 1st Piola stress in the x-direction
    stress_x = c1*(lx - 1/(lx**3 * ly**2)) + c2*(lx * ly**2 - 1/lx**3)
    return stress_x


def Stress_yy(c1, c2, lx, ly):  # planar biaxial 1st Piola stress in the y-direction
    stress_y = c1*(ly - 1/(lx**2 * ly**3)) + c2*(lx**2 * ly - 1/ly**3)
    return stress_y


def objMR(coeffs):  # Mooney Rivlin objective fn
    c1 = coeffs[0]
    c2 = coeffs[1]
    return norm(Stress_xx(c1, c2, lambda_x, lambda_y) - Px) ** 2 + norm(Stress_yy(c1, c2, lambda_x, lambda_y) - Py) ** 2


def objNH(coeff):  # neo Hooke objective fn
    c1 = coeff
    return norm(Stress_xx(c1, 0, lambda_x, lambda_y) - Px) ** 2 + norm(Stress_yy(c1, 0, lambda_x, lambda_y) - Py) ** 2


df_coeffs = pd.DataFrame()  # to save results
df_coeffs['Model'] = ['MR_S1', 'MR_S2', 'MR_S3', 'MR_S4', 'MR_S5', 'MR_S6', 'MR_S7', 'MR_S8',
                      'NH_S1', 'NH_S2', 'NH_S3', 'NH_S4', 'NH_S5', 'NH_S6', 'NH_S7', 'NH_S8']
base_fld = pathlib.Path('/Users/ssp/Desktop/SURI24/SampleData')

for xl_file in base_fld.rglob('*.xlsx'):
    df = pd.read_excel(xl_file)  # input stress-stretch data with 5 different modes
    lambda_x = df['lambda_x']
    lambda_y = df['lambda_y']
    S1_x = df['S1_x']
    S2_x = df['S2_x']
    S3_x = df['S3_x']
    S4_x = df['S4_x']
    S5_x = df['S5_x']
    S6_x = df['S6_x']
    S7_x = df['S7_x']
    S8_x = df['S8_x']
    S1_y = df['S1_y']
    S2_y = df['S2_y']
    S3_y = df['S3_y']
    S4_y = df['S4_y']
    S5_y = df['S5_y']
    S6_y = df['S6_y']
    S7_y = df['S7_y']
    S8_y = df['S8_y']
    Sx = [S1_x, S2_x, S3_x, S4_x, S5_x, S6_x, S7_x, S8_x]
    Sy = [S1_y, S2_y, S3_y, S4_y, S5_y, S6_y, S7_y, S8_y]

    MR_Var1 = np.zeros((8, 1))
    MR_Var2 = np.zeros((8, 1))
    NH_Var1 = np.zeros((8, 1))
    NH_Var2 = np.zeros((8, 1))
    sample_array_NH = []
    for i in range(8):
        Px = Sx[i]
        Py = Sy[i]
        ## Mooney Rivlin model ##
        init = np.ones(2)  # initial guess for coefficients
        bounds = [(0, None), (None, None)]  # positive c1, unrestricted c2
        MRres = minimize(objMR, x0=init, bounds=bounds)  # minimize the objective function and return the optimal coefficients
        MeatName = str(xl_file).rstrip('_results.xlsx').lstrip('/Users/ssp/Desktop/SURI24/').lstrip('SampleData')  # name of file
        print("MR: " + MeatName + '_S#' + str(i+1))
        # print(MRres.x)  # [c1, c2] -- mooney rivlin coefficients
        MR_Var1[i] = round(MRres.x[0], 4)
        MR_Var2[i] = round(MRres.x[1], 4)

        plt.figure(1)
        plt.plot(lambda_x, Px, 'b*')
        plt.plot(lambda_x, Stress_xx(MRres.x[0], MRres.x[1], lambda_x, lambda_y), 'r*')

        plt.figure(2)
        plt.plot(lambda_y, Py, 'b*')
        plt.plot(lambda_y, Stress_yy(MRres.x[0], MRres.x[1], lambda_x, lambda_y), 'r*')

        ## neo Hooke model ##

        init = np.ones(1)  # initial guess

        res = minimize(objNH, x0=init)  # minimize the objective function and return the optimal coefficients
        print("NH: " + MeatName + '_S#' + str(i+1))
        # print(res.x)  # [c1] -- neo Hooke coefficient
        NH_Var1[i] = round(res.x[0], 4)

        # plt.figure(3)
        # plt.plot(lambda_x, Px, 'b*')
        # plt.plot(lambda_x, Stress_xx(res.x, 0, lambda_x, lambda_y), 'r*')
        #
        # plt.figure(4)
        # plt.plot(lambda_y, Py, 'b*')
        # plt.plot(lambda_y, Stress_yy(res.x, 0, lambda_x, lambda_y), 'r*')
        # plt.show()

    Var1_all = np.append(MR_Var1, NH_Var1)
    Var2_all = np.append(MR_Var2, NH_Var2)
    df_coeffs[MeatName+'Var1'] = Var1_all
    df_coeffs[MeatName+'Var2'] = Var2_all

df_coeffs.to_csv('ModelCoefficientsSamplesCORRECTED.csv')  # export results to csv file

