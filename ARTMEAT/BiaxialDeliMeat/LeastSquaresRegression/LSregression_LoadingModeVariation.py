import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pathlib


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
df_coeffs['Model'] = ['MR', 'MR_R2_M1', 'MR_R2_M2', 'MR_R2_M3', 'MR_R2_M4', 'MR_R2_M5',
                      'NH', 'NH_R2_M1', 'NH_R2_M2', 'NH_R2_M3', 'NH_R2_M4', 'NH_R2_M5']
base_fld = pathlib.Path('/Users/ssp/Desktop/SURI24/MeanData')

for xl_file in base_fld.rglob('*.xlsx'):
    df = pd.read_excel(xl_file)  # input stress-stretch data with 5 different modes
    lambda_x = df['lambda_x']
    lambda_y = df['lambda_y']
    Px = df['sigma_xx[kPa]']
    Py = df['sigma_yy[kPa]']

    ## Mooney Rivlin model ##
    init = np.ones(2)  # initial guess for coefficients
    bounds = [(0, None), (None, None)]  # positive c1, unrestricted c2
    MRres = minimize(objMR, x0=init, bounds=bounds)  # minimize the objective function and return the optimal coefficients
    MeatName = str(xl_file).rstrip('_results.xlsx').lstrip('/Users/ssp/Desktop/SURI24/').lstrip('MeanData')  # name of file
    print("MR: " + MeatName)
    print(MRres.x)  # [c1, c2] -- mooney rivlin coefficients

    plt.figure(1)
    plt.plot(lambda_x, Px, 'b*')
    plt.plot(lambda_x, Stress_xx(MRres.x[0], MRres.x[1], lambda_x, lambda_y), 'r*')

    plt.figure(2)
    plt.plot(lambda_y, Py, 'b*')
    plt.plot(lambda_y, Stress_yy(MRres.x[0], MRres.x[1], lambda_x, lambda_y), 'r*')

    ## neo Hooke model ##

    init = np.ones(1)  # initial guess

    res = minimize(objNH, x0=init)  # minimize the objective function and return the optimal coefficients
    print("NH: " + MeatName)
    print(res.x)  # [c1] -- neo Hooke coefficient

    # plt.figure(3)
    # plt.plot(lambda_x, Px, 'b*')
    # plt.plot(lambda_x, Stress_xx(res.x, 0, lambda_x, lambda_y), 'r*')
    #
    # plt.figure(4)
    # plt.plot(lambda_y, Py, 'b*')
    # plt.plot(lambda_y, Stress_yy(res.x, 0, lambda_x, lambda_y), 'r*')
    # plt.show()

    ## R2 scores for each mode in x- and y-direction separately ##
    MR_M1x = round(r2_score(Px[:21], Stress_xx(MRres.x[0], MRres.x[1], lambda_x[:21], lambda_y[:21])), 4)
    MR_M1y = round(r2_score(Py[:21], Stress_yy(MRres.x[0], MRres.x[1], lambda_x[:21], lambda_y[:21])), 4)

    MR_M2x = round(r2_score(Px[21:42], Stress_xx(MRres.x[0], MRres.x[1], lambda_x[21:42], lambda_y[21:42])), 4)
    MR_M2y = round(r2_score(Py[21:42], Stress_yy(MRres.x[0], MRres.x[1], lambda_x[21:42], lambda_y[21:42])), 4)

    MR_M3x = round(r2_score(Px[42:63], Stress_xx(MRres.x[0], MRres.x[1], lambda_x[42:63], lambda_y[42:63])), 4)
    MR_M3y = round(r2_score(Py[42:63], Stress_yy(MRres.x[0], MRres.x[1], lambda_x[42:63], lambda_y[42:63])), 4)

    MR_M4x = round(r2_score(Px[63:84], Stress_xx(MRres.x[0], MRres.x[1], lambda_x[63:84], lambda_y[63:84])), 4)
    MR_M4y = round(r2_score(Py[63:84], Stress_yy(MRres.x[0], MRres.x[1], lambda_x[63:84], lambda_y[63:84])), 4)

    MR_M5x = round(r2_score(Px[84:], Stress_xx(MRres.x[0], MRres.x[1], lambda_x[84:], lambda_y[84:])), 4)
    MR_M5y = round(r2_score(Py[84:], Stress_yy(MRres.x[0], MRres.x[1], lambda_x[84:], lambda_y[84:])), 4)

    NH_M1x = round(r2_score(Px[:21], Stress_xx(res.x[0], 0, lambda_x[:21], lambda_y[:21])), 4)
    NH_M1y = round(r2_score(Py[:21], Stress_yy(res.x[0], 0, lambda_x[:21], lambda_y[:21])), 4)

    NH_M2x = round(r2_score(Px[21:42], Stress_xx(res.x[0], 0, lambda_x[21:42], lambda_y[21:42])), 4)
    NH_M2y = round(r2_score(Py[21:42], Stress_yy(res.x[0], 0, lambda_x[21:42], lambda_y[21:42])), 4)

    NH_M3x = round(r2_score(Px[42:63], Stress_xx(res.x[0], 0, lambda_x[42:63], lambda_y[42:63])), 4)
    NH_M3y = round(r2_score(Py[42:63], Stress_yy(res.x[0], 0, lambda_x[42:63], lambda_y[42:63])), 4)

    NH_M4x = round(r2_score(Px[63:84], Stress_xx(res.x[0], 0, lambda_x[63:84], lambda_y[63:84])), 4)
    NH_M4y = round(r2_score(Py[63:84], Stress_yy(res.x[0], 0, lambda_x[63:84], lambda_y[63:84])), 4)

    NH_M5x = round(r2_score(Px[84:], Stress_xx(res.x[0], 0, lambda_x[84:], lambda_y[84:])), 4)
    NH_M5y = round(r2_score(Py[84:], Stress_yy(res.x[0], 0, lambda_x[84:], lambda_y[84:])), 4)

    df_coeffs[MeatName+'Var1'] = [round(MRres.x[0], 4), MR_M1x, MR_M2x, MR_M3x, MR_M4x, MR_M5x,
                                  round(res.x[0], 4), NH_M1x, NH_M2x, NH_M3x, NH_M4x, NH_M5x]  # save MR and NH results for each meat
    df_coeffs[MeatName + 'Var2'] = [round(MRres.x[1], 4), MR_M1y, MR_M2y, MR_M3y, MR_M4y, MR_M5y, 0, NH_M1y, NH_M2y, NH_M3y,
                                    NH_M4y, NH_M5y]  # save MR and NH results for each meat

df_coeffs.to_csv('ModelCoefficientsCORRECTED.csv')  # export results to csv file

