#!/usr/bin/python3
from numpy import fft
import numpy as np
from scipy import constants as const 

t_au = const.hbar / const.value('Hartree energy')

def ev2au(x):
	return x * const.e  / const.value('Hartree energy') / (2*np.pi)

def pade(signal, length, timestep, freqs, single_point = False):			# Adapted from J. J. Goings' code (https://github.com/jjgoings/pade)
	"""
	Approximate Fourier transform by pade approximants
	"""
	
	N = length//2

	d = -signal[N+1:2*N]

	try:
		from scipy.linalg import toeplitz, solve_toeplitz
		# Instead, form G = (c,r) as toeplitz
		#c = signal[N:2*N-1]
		#r = np.hstack((signal[1],signal[N-1:1:-1]))
		b = solve_toeplitz((signal[N:2*N-1], np.hstack((signal[1],signal[N-1:1:-1]))), d, check_finite=False)

	except (ImportError, np.linalg.linalg.LinAlgError) as e:  
		# OLD CODE: sometimes more stable
		# G[k,m] = signal[N - m + k] for m,k in range(1,N)
		G = signal[N + np.arange(1, N)[:,None] - np.arange(1, N)]
		b = np.linalg.solve(G, d)

	b = np.hstack((1,b))
	a = np.dot(np.tril(toeplitz(signal[0:N])),b)
	p = np.poly1d(a)
	q = np.poly1d(b)

	
	# print(W)
	
	if single_point:
		return p, q
	else:
		W = np.exp(-1j * freqs * timestep * 2 * np.pi)
		return p(W)/q(W)



def Fourier(signal, ts, **kwargs):
	"""
	performs 1 d Fouriertransforms for the input signal

	input: 			- signal(s) as numpy array with shape (number_of_signals, signal_length), e. g. (3, 30000) for x, y, z, components of a dipole signal
					  or (signal_length, ) for just one signal
					- timestep in au
					- method for fouriertransform
					- optional arguments, e. g. broadening

	returns:		frequency, spectrum (complex)	
	"""
	length = signal.shape[1] if len(signal.shape) is not 1 else len(signal)

	# damp signal
	broadening = kwargs['broadening'] # if 'broadening' in kwargs else None
	if broadening is not None:
		print('broadening...', broadening)
		print(ts)
		print(length)
		factor = np.exp(-(ts*np.arange(length)*float(broadening)))*ts

		# if len(signal.shape) == 1:
		# 	signal *= factor

		# else:
		# 	for index, sig in enumerate(signal):
		# 		signal[index] = sig*factor 

		if len(signal.shape) == 1:
			signal_ft = signal*factor

		else:
			signal_ft = np.asarray([sig*factor for sig in signal])

	signal_ft = signal_ft if broadening is not None else signal


	# choose method to perform the Fourier transform
	method = kwargs['method']

	# Fast Fourier transform for real time signal
	if method == 'rfft':

		yield fft.rfftfreq(length, ts)				# yield frequencies in atomic units as first value
		print('FFT...')
		if len(signal_ft.shape) == 1:					# yield rfft if there is just one input signal

			yield  fft.rfft(signal_ft)

		else:										# yield rffts for multiple input signals successively
			for i in signal_ft:

				yield fft.rfft(i)


	# Pade-approximants for Fourier transform (see pade function)
	elif method == 'pade':
		#check whether frequency range and resolution has been specified

		# convert frequencies from [eV] to []
		# if len(kwargs['ws'])==3:
		w_start = ev2au(kwargs['ws'][0])
		w_stop = ev2au(kwargs['ws'][1])
		w_step = ev2au(kwargs['ws'][2])
		freq = np.arange(w_start, w_stop, w_step)

		yield freq 	
										# yield frequencies in atomic units as first value
		print('Pade-approximation...')
		if len(signal_ft.shape) == 1:					# yield pade-FT if there is just one input signal
			yield pade(signal_ft, length, ts, freq)

		else:										# yield pade-FTs for multiple input signals successively
			for i in signal_ft:
				yield pade(i, length, ts, freq)

	elif method == 'pade_poly':
		freq = 0
		yield freq
		print('Pade-approximation...')
		if len(signal_ft.shape) == 1:					# yield pade-FT if there is just one input signal
			yield pade(signal_ft, length, ts, freq, single_point = True)

		else:										# yield pade-FTs for multiple input signals successively
			for i in signal_ft:
				yield pade(i, length, ts, freq, single_point = True)


	else:
		raise KeyError("FT-method unknown")




if __name__ == '__main__':
	import matplotlib.pyplot as plt
	# from scipy.fftpack import fft

	delta_T = 1/800.0
	N = 600
	x = np.linspace(0,delta_T*N,num=N)
	a = np.sin(50*2*np.pi*x)
	b = np.sin(100*2*x*np.pi)
	f = fft.rfftfreq(N, delta_T)
	f2 = fft.fftfreq(N, delta_T)

	FT = Fourier(np.vstack((a,b)), delta_T, broadening=0.5)
	FT2 = Fourier(np.vstack((a,b)), delta_T)
	t = next(FT)
	t2 = next(FT2)
	nodamp_1 = next(FT2)
	nodamp_2 = next(FT2)
	damp_1 = next(FT)
	damp_2 = next(FT)
	test = fft.fft(a)

	plt.plot(f, nodamp_1)
	# plt.plot(f, nodamp_2)
	# plt.plot(f, damp_1)
	# plt.plot(f, damp_2)
	plt.plot(f2, test)

	plt.show()