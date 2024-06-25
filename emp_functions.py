import numpy as np


def lim(function, target, l0, r0): #finds the limit of a function.
    converging = True

    while converging:
        l1, r1 = np.mean([l0, target]), np.mean([r0, target]) # Half the difference between the left/right hand value and the target
        l2, r2 = np.mean([l1, target]), np.mean([r1, target])

        fl0, fr0 = function(l0), function(r0)
        fl1, fr1 = function(l1), function(r2)
        fl2, fr2 = function(l2), function(r2)

        delta_l0, delta_r0 = np.abs(fl1 -fl0), np.abs(fr1 -fr0)
        delta_l1, delta_r1 = np.abs(fl2 -fl1), np.abs(fr2 -fr1)

        if delta_l1 < delta_l0 and delta_r1 < delta_r0: #limit is converging :)
            l0, r0 = l1, r1
        else: #limit is diverging >:(. This is assumed to be a numerical error rather than a mathematical property of the function.
            converging = False

    return np.mean([fl0, fr0]) #Given that the limit converges, it should lie somewhere between the left and right hand limits.

def recenter(interferogram): #moves the (positive) peak to the center. 
    length = len(interferogram)
    max_index = np.argmax(interferogram)
    tau = length//2 -max_index

    FT = np.fft.fft(interferogram)
    freq = np.fft.fftfreq(length)
    FT *= np.exp(-2j*np.pi*freq*tau)

    interferogram = np.fft.ifft(FT)
    return interferogram

def kramers_kronig(omega, rho): # omega is the angular frequency. rho is the absolute part of the spectrum. (square root of the power spectrum)
    assert len(omega) == len(rho), "All values in the function must have a corrisponding frequency."
    N = len(omega)

    sort = np.argsort(omega)
    unsort = np.argsort(sort)
    delta_omega = np.diff(omega[sort], append= 2*omega[sort][-1] -omega[sort][-2]) # Extrapolate the last value.
    delta_omega = delta_omega[unsort] # When approximatating an intergral to the sum of many rectangles, we must find the area by multiplying by the width of the rectangles.

    summation = np.zeros(N)
    integrand = np.zeros(N)

    for x, dx, rho_x, n in zip(omega, delta_omega, rho, np.arange(N)):
        numerator = -omega *np.log(rho_x /rho)
        denominator = omega**2 -x**2
        integrand= numerator/denominator

        ###### SOLVING LIMIT #####
        rho_func = lambda y: np.interp(y, omega, rho)
        integrand_func = lambda y: -x *np.log(rho_func(y)/rho_x) /(x**2 -y**2) # I have used x as the frequency and y as the integral variable. (instead of omega and x)

        integrand[n] = lim(integrand_func, x, x-dx, x+dx)

        summation += integrand *dx

    return 2/np.pi *summation #phase

def gerchberg_saxon(rho, sensitivity_mask= None, initial_guess= None, iterations= 10000, noise_level= 0.01, beta= 1, gamma= 0.95): #rho is the absolute part of the spectrum. The Gerchberg-Saxon algorithm is not analytical unlike Kramers-Kronig.
    
    if initial_guess is None:
        array_length = 2*(len(rho) -1)
        initial_guess = np.zeros(array_length, dtype= np.float64)
    else:
        array_length = len(initial_guess)

    if sensitivity_mask is None: sensitivity_mask = np.full_like(rho, True, dtype= bool)

    #initialise loop
    IFFT0 = np.copy(initial_guess)

    #begin loop
    for n in range(iterations):

        FT0 = np.fft.rfft(IFFT0, norm= "forward")

        ## FOURIER DOMAIN CONSTRAINT
        phase = np.angle(FT0)
        FT1 = np.copy(FT0)
        FT1[sensitivity_mask] = rho[sensitivity_mask] *np.exp(1j *phase[sensitivity_mask])
        complex_form_factor = np.copy(FT1) # OUTPUT OF ALGORITHM IS HERE
        FT1[~sensitivity_mask] *= gamma # suppress unknown frequencies. The gerchberg-saxon algorithm's biggest strength and weakness is how it can guess unknown frequencies. This often leads to a lot of noise. Because we are expecting a gaussian bunch, it may be better to multiply by a half gaussian.

        IFFT1 = np.fft.irfft(FT1, n= array_length, norm= "forward")

        ## SUPPORT CONSTRAINT
        is_causal = np.full(array_length, True, dtype= bool)
        is_causal[array_length//2:] = np.abs(IFFT1[array_length//2:]) < noise_level
        violates_constraint = np.logical_not(is_causal)

        ## apply the support constraint
        IFFT0[~violates_constraint] = IFFT1[~violates_constraint]
        IFFT0[violates_constraint] = IFFT0[violates_constraint] -beta*IFFT1[violates_constraint] # Fienup's application of the support constraint
        #IFFT0[violates_constraint] = (IFFT0[violates_constraint] +IFFT1[violates_constraint]) /2 # this scheme converges well but often gets stuck in a local minima

    return complex_form_factor #complex form factor #np.angle(FT1) #phase