from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl
import matplotlib.pyplot as plt
import numpy as np

kernel_path = "path to kernel"

session=WolframLanguageSession(kernel_path)

def bivariate_fox_h(k, q, p, N, a, rho, lambda_val, A, b, c, gamma_bar, gamma_0):
    x_val = 1 / (lambda_val * A * np.sqrt((q**(N-k)) * gamma_0))
    y_val = (1 / ((b**c) * (A**c))) * (np.sqrt(1 / (q**k * gamma_bar))**c)

    # Use wl.Power for variables inside the integral
    h_val = session.evaluate(wl.With(
        {'x': x_val, 'y': y_val, 'p': p, 'N': N, 'k': k, 'c': c, 'rho': rho, 'a': a},
        wl.NIntegrate(
            (wl.Gamma(1 - wl.p - ((wl.N - wl.k)/2)*s - ((wl.k*wl.c)/2)*t) *
             (wl.Gamma(1 + s) * wl.Gamma(wl.rho**2 + 1 + s)) / 
             (wl.Gamma(1 - s) * wl.Gamma(wl.rho**2 - s) * wl.Gamma(0 - s)) *
             (wl.Gamma(1 + t) * wl.Gamma(wl.rho**2/wl.c + 1 + t)) / 
             (wl.Gamma(wl.a - t) * wl.Gamma(wl.rho**2/wl.c - t) * wl.Gamma(0 - t)) *
             wl.Power(wl.x, s) * wl.Power(wl.y, t)), # Replaced ** with wl.Power
            [s, wl.Complex(0.1, -20), wl.Complex(0.1, 20)],
            [t, wl.Complex(0.1, -20), wl.Complex(0.1, 20)],
            PrecisionGoal=6, WorkingPrecision=30
        )
    ))
    return h_val

def calculate_pe_bar(q, p, N, omega, a, rho, lambda_val, A, b, c, gamma_bar, gamma_0):
    # 1. Leading coefficient - Fully evaluate in Mathematica
    p1 = session.evaluate(
        wl.Divide(
            wl.Times(q, wl.Power(rho, 2 * N)),
            wl.Times(2, wl.Power(q, p), wl.Gamma(p))
        )
    )

    total_sum = 0
    for k in range(N + 1):
        # 2. term_coeff - Combined into one evaluate call
        term_coeff = session.evaluate(
            wl.Times(
                wl.Binomial(N, k),
                wl.Power(omega, N - k),
                wl.Power(
                wl.Divide(
                    wl.Minus(1, omega),
                    wl.Times(c, wl.Gamma(a))
                ),
                k
            )
        )
    )

        # 3. First Meijer G term
        z1 = 1 / (lambda_val * A * np.sqrt(1 / gamma_bar))
        p2a = session.evaluate(
            wl.Power(
                wl.MeijerG([[1, wl.Power(rho, 2) + 1], []], [[1, wl.Power(rho, 2), 0], []], z1),
                N - k - 1
            )
        )

        # 4. Second Meijer G term
        # Note: using wl.Power for the constants to ensure precision
        z2 = session.evaluate(
            (1 / (wl.Power(b, c) * wl.Power(A, c))) * wl.Power(wl.Sqrt(1 / gamma_bar), c)
        )
        p2b = session.evaluate(
            wl.Power(
                wl.MeijerG([[1, (wl.Power(rho, 2) / c) + 1], []], [[a, (wl.Power(rho, 2) / c), 0], []], z2),
                k - 1
            )
        )

        # 5. Bivariate Fox H term
        p2c = bivariate_fox_h(k, q, p, N, a, rho, lambda_val, A, b, c, gamma_bar, gamma_0)

        # Now all variables (term_coeff, p2a, p2b, p2c) are Python floats
        total_sum += term_coeff * p2a * p2b * p2c

    return p1 * total_sum


s = wl.Symbol('s')
t = wl.Symbol('t')

A = 0.1639
rho = 0.9875
p = 1/2
q = 1/2
gamma_0 = 10
gamma_bar = 26
omega = 0.5117
lambda_val = 0.1602
a=0.0075
b=2.9963
c=216.8356
N=1

res = calculate_pe_bar(q,p,N,omega,a,rho,lambda_val,A,b,c,gamma_bar,gamma_0)

session.terminate()

print(res)




