#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sympy as sp


def main():
    """Main function to define sympy variables for polyconvexity analysis"""
    lambda1 = sp.Symbol('lambda1', real=True, positive=True)
    lambda2 = sp.Symbol('lambda2', real=True, positive=True)
    lambda3 = sp.Symbol('lambda3', real=True, positive=True)
    x = sp.Symbol('x', real=True, positive=True)
    ## Stiffness parameters
    mu_1 = 38.3
    mu_2 = 7.04
    mu_3 = 52.0
    ## Exponents
    m = 3.32
    alpha = 5.88
    ## Invariants
    J = lambda1 * lambda2 * lambda3
    I1 = lambda1 ** 2 + lambda2 ** 2 + lambda3 ** 2
    I2 = lambda1 ** 2 * lambda2 ** 2 + lambda2 ** 2 * lambda3 ** 2 + lambda3 ** 2 * lambda1 ** 2
    I1_bar = I1 / J**(2/3) - 3
    I2_bar = I2 / J**(4/3) - 3
    ## Strain energy
    w = mu_1 * (I1_bar - 3) + mu_2 * (J**m - m * sp.log(J) - 1) + mu_3 * J**alpha * (I1_bar - 3)
    ## Polyconvexity condition
    wpp = sp.simplify(sp.diff(sp.diff(w, lambda1), lambda1) * lambda1 ** 2 )
    wpp.subs({lambda1: 1.0, lambda2: 1.0, lambda3: 1.0})
    print(wpp.subs({lambda1: x, lambda2: 1.0, lambda3: 1.0}))
    print(wpp.subs({lambda1: 1.0, lambda2: x, lambda3: 1.0}))

if __name__ == "__main__":
    main()

