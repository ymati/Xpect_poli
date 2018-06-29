from Xpect_new import Xtract
from Xpect_new import convert_units as conv
import numpy as np 
from scipy import constants as const
import re
from Xpect_new import XFT_2 as XFT
import math

def find_nearest(array,value):
	"""
	find nearest entry in a sorted array and return its index and value
	"""
	idx = np.searchsorted(array, value, side="left")
	if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
	    return idx-1, array[idx-1]
	else:
	    return idx, array[idx]


def string2ts(string):						
	"""
	converts input string to SI float for different units of the timestep, e. g. '0.025 fs' --> 2.5e-17 [s]
	"""
	if string.find('fs') > 0:
		timestep = conv.fs2au(float(re.findall("\d*\.\d+|\d+\.?\d*", string)[0]))
	elif string.find('au') > 0:
		timestep = float(re.findall("\d*\.\d+|\d+\.?\d*", string)[0])
	else:
		raise KeyError("units of timestep unknown")
	return timestep

def shift2upperplane(spectrum):
	r = np.sqrt(np.real(spectrum)**2 + np.imag(spectrum)**2)
	phi = np.abs(np.arctan2(np.imag(spectrum), np.real(spectrum)))

	spectrum_shift = np.vectorize(complex)(r*np.cos(phi),r*np.sin(phi))

	return spectrum_shift


def standardize_signals(args):

	if args['code'] not in ["CP2K", "NWC"]:
		raise KeyError("Quantum chemistry code not implemented with RT-TDDFT yet")
	print('reading data files...')

	sign = [Xtract.Xtract(path = i, code = args['code'], properties = args['properties'] ).extract() for i in args['files']]
	# axes are [file, property, timestep, values (i.e. value or vector)]

	min_length = len(sign[0][0])

	# cut signals to have the same ammount of timesteps
	if any(len(j) != min_length for i in sign for j in i):
		print('The length of the different dipole signals do not match and will be adapted to the shortest: ')
		min_length = min([len(j) for i in sign for j in i])
		print(min_length)

		sign_short = np.empty([len(sign), len(args['properties']), min_length, 3])
		for i, prop in enumerate(sign):
			for j, signal in enumerate(prop):
				mask = [True if i < min_length else False for i in range(len(signal))]
				sign_short[i][j] = signal[mask]
		sign = sign_short

	sign = np.asarray(sign)
	
	if len(args['files']) == 3:
		if args['diag']:
			data = np.swapaxes(np.dstack((sign[0,:, :,0], sign[1,:,:,1], sign[2,:,:,2])), 1, 2) 

		else:
			data = np.swapaxes(np.dstack((sign[0,:, :,:], sign[1,:,:,:], sign[2,:,:,:])), 1, 2)		# now axes are [property, values(xx,xy,xz...), timestep]
	else:
		print(sign.shape)
		sign.shape=(1,min_length,3)
		data = np.swapaxes(sign, 1, 2)
		print(data.shape)

	return {i : data[index] for index, i in enumerate(args['properties'])}, min_length


def parse_ft_args(args, signal_length):
	"""
	parse optional parameters for the Fourier-transform and set default values
	"""
	if 'start' not in args:
		args['start'] = 0 

	if 'stop' in args:
		# print(signal_length, 'signal_length')
		# print(args['stop'], 'stop value')
		if args['stop'] > signal_length:
			raise ValueError("Signal is not long enough for requested stop value")
	else:
		args['stop'] = signal_length

	if 'broadening' not in args:
		args['broadening'] = None

	if 'FT_method' not in args:
		args['FT_method'] = 'rfft'

	if 'pade_ws' not in args:
		args['pade_ws'] = None

	return args


def diff_3(a, b, delta):
	"""
	Three point differentiantion formula [f(x+dx) - f(x-dx)]/2dx
	"""
	return (a - b)/float(2*delta)


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

	if single_point:
		return p, q
	else:
		W = np.exp(-1j * freqs * timestep * 2 * np.pi)
		return p(W)/q(W)


def gss(f, a, b, tol=np.finfo(float).eps):
    '''
    golden section search
    to find the maximum of f on [a,b]
    f: a strictly unimodal function on [a,b]

    example:
    >>> f = lambda x: (x-2)**2
    >>> x = gss(f, 1, 5)
    >>> x
    2.000009644875678

    '''
    # golden section search
    gr = (math.sqrt(5) + 1) / 2

    c = b - (b - a) / gr
    d = a + (b - a) / gr 
    while abs(c - d) > tol:
        if f(c) > f(d):
            b = d
        else:
            a = c

        # we recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2


###################################################################################################
#####	Read and FT-class for reading the signals and Fourier-transform them			      #####
###################################################################################################

class randft:
	def __init__(self, args):
		self.timestep = string2ts(args['ts'])
		print('Timestep:\t', self.timestep, 'a. u.')

		self.signals, self.signal_length = standardize_signals(args)

		# get initial dipole moments
		self.p_0 = {k : tuple(j[0] for j in self.signals[k]) for k in self.signals}

		# center signals
		for k in self.signals:
			for index, trajectory in enumerate(self.signals[k]):
				self.signals[k][index] = trajectory - trajectory[0]

	def ft(self, kwargs):
		# print(self.signal_length, 'ft')
		ft_args = parse_ft_args(kwargs, self.signal_length)
		start = ft_args['start']
		stop = ft_args['stop']

		# dim = self.signals[kwargs['property']].shape[0]

		signals = self.signals[kwargs['property']]

		if ft_args['FT_method'] is not 'pade_poly':
			FT = XFT.Fourier(signals[:,start:stop], self.timestep, method = ft_args['FT_method'], broadening = ft_args['broadening'], ws = ft_args['pade_ws'])
			# convert frequencies from a. u. to Hz 
			yield next(FT)*const.value('Hartree energy')/const.hbar

		else:

			FT = XFT.Fourier(signals[:,start:stop], self.timestep, method = 'pade_poly', broadening = ft_args['broadening'], ws = ft_args['pade_ws'])
			yield next(FT) # empty frequency value
		# frequencies = next(FT)*const.value('Hartree energy')/const.hbar
		

		# spectrum = np.asarray([i for i in FT])
		if ft_args['FT_method'] is not 'pade_poly':
			yield np.asarray([i for i in FT])
		else:
			yield [i for i in FT]
		# return frequencies, spectrum



###################################################################################################
#####				property classes: calculate properties from FT-transformed signals		  #####
###################################################################################################

class polarizability(randft):
	def __init__(self, kwargs):
		self.prop = 'RT-edipole'
		self.diag = kwargs['diag']
		kwargs['properties'] = (self.prop, )
		super().__init__(kwargs)
		# print("V2")


	def tensor(self, kwargs):
		kwargs['property'] = self.prop
		kwargs['kappa'] = float(1)

		f, FT = self.ft(kwargs)
		if self.diag:
			if kwargs['FT_method'] == 'pade_poly':
				return f, FT
			else:			
				return f, FT.reshape((3, len(f)))/kwargs['kappa']
		else:
			if kwargs['FT_method'] == 'pade_poly':
				return f, FT.reshape((3,2,1))
			else:
				return f, FT.reshape((3, 3, len(f)))/kwargs['kappa']

class beta(randft):
	def __init__(self, kwargs):
		self.prop = 'RT-mdipole'
		self.diag = kwargs['diag']
		kwargs['properties'] = (self.prop, )
		super().__init__(kwargs)

	def tensor(self, kwargs):
		kwargs['property'] = self.prop
		kwargs['kappa'] = float(1)

		f, FT = self.ft(kwargs)
		if self.diag:			
			return f[1:], 1j * const.c /  f[1:] * FT.reshape((3, len(f)))[:,1:]/kwargs['kappa']
		else:
			return f[1:], 1j * const.c /  f[1:] * FT.reshape((3, 3, len(f)))[:,:,1:]/kwargs['kappa']

#		1j * const.c /  self.frequencies[1:] *self.b_diag[:,1:]


class FT(randft):
	def __init__(self, kwargs):
		kwargs_read = {'ts' : kwargs['ts'], 'code' : kwargs['code'], 'files' : kwargs['files'], 'diag' : False, 'properties' : ('RT-edipole', ) } 

		self.read = randft(kwargs_read)

		self.timestep = self.read.timestep
		self.signal_length = self.read.signal_length

	def get_polarizabilities(self, kwargs):
		ft_args = parse_ft_args(kwargs, self.signal_length)
		start = ft_args['start']
		stop = ft_args['stop']
		ft_length = stop - start

		# damping the signal
		if kwargs['broadening'] is not None:
			print('damping signal ...')
			factor = np.exp(-(self.timestep*np.arange(ft_length)*float(kwargs['broadening'])))*self.timestep
			signal = np.asarray([conv.debye2au(sig)*factor for sig in self.read.signals['RT-edipole'][:,start:stop]])
			self.signal_damped = signal
		else:
			signal = debye2au(self.read.signals['RT-edipole'][:,start:stop])

		# do Fourier transform
		# Real fast Fourier transform
		if kwargs['FT_method'] == 'rfft':
			print('rfft ...')
			self.single_point = False
			# print(self.timestep)
			# print(ft_length)
			self.frequencies = np.fft.rfftfreq(ft_length, self.timestep)*const.value('Hartree energy')/const.hbar
			self.ft = [np.fft.rfft(sig) for sig in signal]

		# Pade approximation
		elif kwargs['FT_method'] == 'pade':
			print('Pade approximants...')
			if isinstance(kwargs['pade_ws'], tuple):
				self.single_point = False
				w_start = conv.ev2au(kwargs['pade_ws'][0])
				w_stop = conv.ev2au(kwargs['pade_ws'][1])
				w_step = conv.ev2au(kwargs['pade_ws'][2])
				freq_pade = np.arange(w_start, w_stop, w_step)
				self.frequencies = freq_pade *const.value('Hartree energy')/const.hbar
				self.ft = tuple([pade(sig, ft_length, self.timestep, freq_pade, single_point = False) for sig in signal])
				# print('tuple')

			else:
				self.single_point = True
				self.ft = tuple([pade(sig, ft_length, self.timestep, 0, single_point = True) for sig in signal])

		else:
			raise KeyError('FT-method unknown')

	def FT(self, kwargs):
		return self.frequencies, np.real(self.ft), np.imag(self.ft)

###################################################################################################
#####				Spectrum classes: have a spectrum method that returns (\omega, S)	      #####
###################################################################################################

class absorption:
	def __init__(self, kwargs):
		kwargs['diag'] = True
		self.polarizability = polarizability(kwargs)

	def spectrum(self, kwargs):
		freq, trace = self.polarizability.tensor(kwargs)
		# print(freq, trace)
		# if not kwargs["FT_method"] == 'pade_poly':
		trace = shift2upperplane(trace)
		return freq, 4 * np.pi * freq / (3 * const.c) * np.imag(trace[0] + trace[1] + trace[2])


class ECD:
	def __init__(self, kwargs):
		kwargs['diag'] = True
		self.beta = beta(kwargs)

	def spectrum(self, kwargs):
		freq, trace = self.beta.tensor(kwargs)
		return freq, 3 * freq / ( np.pi * const.c) * np.imag(trace[0] + trace[1] + trace[2])

class absorption_2:
	def __init__(self, kwargs):
		kwargs_read = {'ts' : kwargs['ts'], 'code' : kwargs['code'], 'files' : kwargs['files'], 'diag' : True, 'properties' : ('RT-edipole', ) } 

		self.read = randft(kwargs_read)

		self.timestep = self.read.timestep
		self.signal_length = self.read.signal_length

	def get_polarizabilities(self, kwargs):
		ft_args = parse_ft_args(kwargs, self.signal_length)
		start = ft_args['start']
		stop = ft_args['stop']
		ft_length = stop - start

		# damping the signal
		if kwargs['broadening'] is not None:
			print('damping signal ...')
			factor = np.exp(-(self.timestep*np.arange(ft_length)*float(kwargs['broadening'])))*self.timestep
			signal = np.asarray([conv.debye2au(sig)*factor for sig in self.read.signals['RT-edipole'][:,start:stop]])
			self.signal_damped = signal
		else:
			signal = debye2au(self.read.signals['RT-edipole'][:,start:stop])

		# do Fourier transform
		# Real fast Fourier transform
		if kwargs['FT_method'] == 'rfft':
			print('rfft ...')
			self.single_point = False
			# print(self.timestep)
			# print(ft_length)
			self.frequencies = np.fft.rfftfreq(ft_length, self.timestep)*const.value('Hartree energy')/const.hbar
			self.ft = [np.fft.rfft(sig) for sig in signal]

		# Pade approximation
		elif kwargs['FT_method'] == 'pade':
			print('Pade approximants...')
			if isinstance(kwargs['pade_ws'], tuple):
				self.single_point = False
				w_start = conv.ev2au(kwargs['pade_ws'][0])
				w_stop = conv.ev2au(kwargs['pade_ws'][1])
				w_step = conv.ev2au(kwargs['pade_ws'][2])
				freq_pade = np.arange(w_start, w_stop, w_step)
				self.frequencies = freq_pade *const.value('Hartree energy')/const.hbar
				self.ft = tuple([pade(sig, ft_length, self.timestep, freq_pade, single_point = False) for sig in signal])
				# print('tuple')

			else:
				self.single_point = True
				self.ft = tuple([pade(sig, ft_length, self.timestep, 0, single_point = True) for sig in signal])

		else:
			raise KeyError('FT-method unknown')		

	def spectrum(self, kwargs):
		kappa = kwargs['kappa'] if 'kappa' in kwargs else float(1)
		trace = shift2upperplane(np.asarray(self.ft).reshape((3, len(self.frequencies))))
		return conv.f2eV(self.frequencies), 4 * np.pi * self.frequencies / (3 * const.c * kappa) * np.imag(trace[0] + trace[1] + trace[2])

	def _scan(self, w):
		W = np.exp(-1j * conv.ev2au(w) * self.timestep * 2 * np.pi)
		trace = shift2upperplane([i(W)/j(W) for (i, j) in self.ft])

		f = w*const.e/const.h 				# convert to Hz [SI]
		return 4 * np.pi * f / (3 * const.c) * np.imag(trace[0] + trace[1] + trace[2])		

	def mean_polarizability(self, w, kappa):
		W = np.exp(-1j * conv.ev2au(w) * self.timestep * 2 * np.pi)
		alpha = np.asarray([i(W)/j(W) for (i, j) in self.ft])/kappa
		return alpha

	def exc_ener(self, lower, upper):
		return gss(self._scan, lower, upper)

###################################################################################################
#####		Raman spectroscopy: calculate crossection for one normal mode 				      #####
###################################################################################################

class Raman:
	def __init__(self, kwargs):

		kwargs_p = {'ts' : kwargs['ts'], 'code' : kwargs['code'], 'files' : kwargs['files_p'], 'diag' : False, 'properties' : ('RT-edipole', ) } 
		kwargs_m = {'ts' : kwargs['ts'], 'code' : kwargs['code'], 'files' : kwargs['files_m'], 'diag' : False, 'properties' : ('RT-edipole', ) } 

		self.read_p = randft(kwargs_p)   		# read in dipole signals for geometry displaced along positive normal mode direction
		self.read_m = randft(kwargs_m)     		# read in dipole signals for geometry displaced along negative normal mode direction

		# print(self.read_p.signals)
		# print(self.read_m.signals)

		self.timestep = self.read_p.timestep
		self.signal_length = min(self.read_p.signal_length, self.read_m.signal_length)

	def get_polarizabilities(self, kwargs):
		ft_args = parse_ft_args(kwargs, self.signal_length)
		start = ft_args['start']
		stop = ft_args['stop']
		ft_length = stop - start

		if 'fieldstrength' in kwargs:
			self.fieldstrength = kwargs['fieldstrength']
		else:
			self.fieldstrength = 1

		print('fieldstrength [a. u.]:', self.fieldstrength)

		# damping the signal
		if kwargs['broadening'] is not None:
			print('damping signal ...', kwargs['broadening'] ,' a. u.')
			factor = np.exp(-(self.timestep*np.arange(ft_length)*float(kwargs['broadening'])))*self.timestep
			signal_p = np.asarray([conv.debye2au(sig)*factor for sig in self.read_p.signals['RT-edipole'][:,start:stop]])
			signal_m = np.asarray([conv.debye2au(sig)*factor for sig in self.read_m.signals['RT-edipole'][:,start:stop]])
			self.signal_damped = signal_m
		else:
			signal_p = debye2au(self.read_p.signals['RT-edipole'][:,start:stop])
			signal_m = debye2au(self.read_m.signals['RT-edipole'][:,start:stop])


		# do Fourier transform
		# Real fast Fourier transform
		if kwargs['FT_method'] == 'rfft':
			print('rfft ...')
			self.single_point = False
			# print(self.timestep)
			# print(ft_length)
			self.frequencies = np.fft.rfftfreq(ft_length, self.timestep)*const.value('Hartree energy')/const.hbar
			self.ft_p = [np.fft.rfft(sig_p)/self.fieldstrength for sig_p in signal_p]
			self.ft_m = [np.fft.rfft(sig_m)/self.fieldstrength for sig_m in signal_m]

		# Pade approximation
		elif kwargs['FT_method'] == 'pade':
			print('Pade approximants...')
			if isinstance(kwargs['pade_ws'], tuple):
				self.single_point = False
				w_start = conv.ev2au(kwargs['pade_ws'][0])
				w_stop = conv.ev2au(kwargs['pade_ws'][1])
				w_step = conv.ev2au(kwargs['pade_ws'][2])
				freq_pade = np.arange(w_start, w_stop, w_step)
				self.frequencies = freq_pade *const.value('Hartree energy')/const.hbar
				self.ft_p = tuple([pade(sig, ft_length, self.timestep, freq_pade, single_point = False)/self.fieldstrength for sig in signal_p])
				self.ft_m = tuple([pade(sig, ft_length, self.timestep, freq_pade, single_point = False)/self.fieldstrength for sig in signal_m])
				# print('tuple')

			else:
				self.single_point = True
				self.ft_p = tuple([pade(sig, ft_length, self.timestep, 0, single_point = True) for sig in signal_p])
				self.ft_m = tuple([pade(sig, ft_length, self.timestep, 0, single_point = True) for sig in signal_m])

		else:
			raise KeyError('FT-method unknown')

	def deriv(self, diff, freq): 		# take derivative at a specific excitation frequency
		if self.single_point:
			# freq = 0.01 #!!!
			W = np.exp(-1j * conv.ev2au(freq) * self.timestep * 2 * np.pi)
			self.resonancefreq =  conv.eV2f(freq) 
			self.a_p = np.asarray([i(W)/j(W)/self.fieldstrength for (i, j) in self.ft_p]).reshape((3,3))
			self.a_m = np.asarray([i(W)/j(W)/self.fieldstrength for (i, j) in self.ft_m]).reshape((3,3))
		else:
			# get nearest discretized frequency to the excitation frequency
			exc_freq_Hz = conv.eV2f(freq)
			index, self.resonancefreq = find_nearest(self.frequencies, exc_freq_Hz)
			# print(index,self.resonancefreq)
			self.a_p = np.asarray([i[index] for i in self.ft_p]).reshape((3,3))
			self.a_m = np.asarray([i[index] for i in self.ft_m]).reshape((3,3))
				
			# if np.abs(self.resonancefreq - exc_freq_Hz) > 1e10:
			# 	print('Due to the discrete signal the excitation frequency was shifted from', freq,'eV to', f2eV(self.resonancefreq), 'eV')
		print('polarizability_p: ', self.a_p)
		print('polarizability_m: ', self.a_m)
		print('polarizability_m_SI: ', conv.polarizability_au2SI(self.a_m))
		print('mean polarizability_p: ', 1/3*(self.a_p[0,0] + self.a_p[1,1] + self.a_p[2,2]))
		print('mean polarizability_m: ', 1/3*(self.a_m[0,0] + self.a_m[1,1] + self.a_m[2,2]))
		print('difference: ', self.a_p - self.a_m)
		print('diff', diff)
		self.da = diff_3(self.a_p, self.a_m, diff)

		self.da = conv.polarizability_deriv_au2SI(self.da / np.sqrt(const.m_e/const.value('atomic mass constant')) ) / np.sqrt(const.m_e) #/(1e-10/const.value('Bohr radius')) 
		# /np.sqrt(const.m_e/const.value('atomic mass constant')) 		to convert denominator to atomic units <-- mass weighted coordinates are done using atomic mass units 1 u = const.value('atomic mass constant')
		# /np.sqrt(const.m_e) 											to convert massweighted coordinates to SI units
		# /(1e-10/const.value('Bohr radius'))							to convert to SI units, for the displacements units of angstrom were used
		print(self.da)


	def intensity(self, kwargs):
		nu_p = kwargs['nu_p']

		ak = 1/float(3) * np.abs( self.da[0,0] + self.da[1,1] + self.da[2,2])

		gamma_k2 = 1/float(2) * ( np.abs(self.da[0,0] - self.da[1,1])**2 + np.abs(self.da[1,1] - self.da[2,2])**2 + np.abs(self.da[2,2] - self.da[0,0])**2  + 6 * ( np.abs(self.da[0,1])**2 + np.abs(self.da[1,2])**2 + np.abs(self.da[2,0])**2 ) )
		# gamma_k2 = 1/float(2) * ( (self.da[0,0] - self.da[1,1])**2 + (self.da[1,1] - self.da[2,2])**2 + (self.da[2,2] - self.da[0,0])**2  + 6 * ( self.da[0,1])**2 + self.da[1,2]**2 + self.da[2,0]**2  )

		# print(f2nu(self.resonancefreq))
		d_sigma = np.pi**2/(const.epsilon_0**2) * (conv.f2nu_SI(self.resonancefreq) - nu_p*100)**4 * const.h / (8 * np.pi**2 * const.c * nu_p*100) * (45*ak**2 + 7 * gamma_k2)/45 * 1 / (1 - np.exp(- const.h * const.c * nu_p*100 / ( const.k * kwargs['T'] )))

		return nu_p, d_sigma, conv.f2eV(self.resonancefreq)

###################################################################################################
#####		RT-TDDFT Excited state gradient method										      #####
###################################################################################################


class ESGM:
	def __init__(self, kwargs):
		kwargs['diag'] = True
		self.polarizability = polarizability(kwargs)

	def pade_poly(self, kwargs):
		kwargs['FT_method'] = 'pade_poly'
		_f, self.polys = self.polarizability.tensor(kwargs)
		# return self.polys

	def single_point(self, w):
		W = np.exp(-1j * conv.ev2au(w) * self.polarizability.timestep * 2 * np.pi)
		
		trace = [i(W)/j(W) for (i, j) in self.polys]
		# print(trace)
		trace = shift2upperplane(trace)
		f = w*const.e/const.h
		return 4 * np.pi * f / (3 * const.c) * np.imag(trace[0] + trace[1] + trace[2])


class ESGM_2:
	def __init__(self, kwargs):
		kwargs['diag'] = True
		kwargs_p = {'ts' : kwargs['ts'], 'code' : kwargs['code'], 'files' : kwargs['files_p'], 'diag' : True, 'properties' : ('RT-edipole', ) } 
		kwargs_m = {'ts' : kwargs['ts'], 'code' : kwargs['code'], 'files' : kwargs['files_m'], 'diag' : True, 'properties' : ('RT-edipole', ) } 

		self.read_p = randft(kwargs_p)		# read dipole signals
		self.read_m = randft(kwargs_m)		# read dipole signals

		print(self.read_p.signals['RT-edipole'].shape)

		self.timestep = self.read_p.timestep
		self.signal_length = min(self.read_p.signal_length, self.read_m.signal_length)	
		print(self.signal_length)

	def get_polarizabilities(self, kwargs):
		self.single_point = True
		ft_args = parse_ft_args(kwargs, self.signal_length)
		start = ft_args['start']
		stop = ft_args['stop']
		ft_length = stop - start

		if 'fieldstrength' in kwargs:
			self.fieldstrength = kwargs['fieldstrength']
		else:
			self.fieldstrength = 1

		print('fieldstrength [a. u.]:', self.fieldstrength)
		# damping the signal
		if kwargs['broadening'] is not None:
			print('damping signal ...')
			factor = np.exp(-(self.timestep*np.arange(ft_length)*float(kwargs['broadening'])))*self.timestep
			signal_p = np.asarray([sig*factor for sig in self.read_p.signals['RT-edipole'][:,start:stop]])
			signal_m = np.asarray([sig*factor for sig in self.read_m.signals['RT-edipole'][:,start:stop]])
			self.signal_damped = signal_m
		else:
			signal_p = self.read_p.signals['RT-edipole'][:,start:stop]
			signal_m = self.read_m.signals['RT-edipole'][:,start:stop]

		# Pade approximation
		if kwargs['FT_method'] == 'pade':
			print('Pade approximants...')

			self.ft_p = tuple([pade(sig, ft_length, self.timestep, 0, single_point = True) for sig in signal_p])
			self.ft_m = tuple([pade(sig, ft_length, self.timestep, 0, single_point = True) for sig in signal_m])

		else:
			raise KeyError('RT-ESGM is only possible with pade approximants, because the discrete FT has usually not enough points to resolve the excitation energies')

	def scan_p(self, w):
		W = np.exp(-1j * conv.ev2au(w) * self.timestep * 2 * np.pi)
		trace = shift2upperplane([i(W)/j(W)/self.fieldstrength for (i, j) in self.ft_p])

		f = w*const.e/const.h 				# convert to Hz [SI]
		return 4 * np.pi * f / (3 * const.c) * np.imag(trace[0] + trace[1] + trace[2])		
	
	def scan_m(self, w):
		W = np.exp(-1j * conv.ev2au(w) * self.timestep * 2 * np.pi)
		trace = shift2upperplane([i(W)/j(W)/self.fieldstrength for (i, j) in self.ft_m])

		f = w*const.e/const.h 				# convert to Hz [SI]
		return 4 * np.pi * f / (3 * const.c) * np.imag(trace[0] + trace[1] + trace[2])		

	def exc_ener_p(self, lower, upper):
		return gss(self.scan_p, lower, upper)

	def exc_ener_m(self, lower, upper):
		return gss(self.scan_m, lower, upper)

	def gradient(self, lower, upper, fd):
		return np.abs(self.exc_ener_p(lower, upper) - self.exc_ener_m(lower, upper))/fd

if __name__ == "__main__":
	# import Xpect.Xpect as Xpect
	# from Xpect import RT_3
	import matplotlib.pyplot as plt

	# broadening =  0.0037	# broadining factor gamma:  exp(- gamma * t) [a. u.]

	# # files_mag = ("/home/jmatti/projects/methyloxirane/ts_au_TZVP/r-methyloxirane_pbe_x-dir_TZVP-output-moments.dat", "/home/jmatti/projects/methyloxirane/ts_au_TZVP/r-methyloxirane_pbe_y-dir_TZVP-output-moments.dat", "/home/jmatti/projects/methyloxirane/ts_au_TZVP/r-methyloxirane_pbe_z-dir_TZVP-output-moments.dat")
	# # files_TZV2P_onerun_r = ("/home/jmatti/projects/methyloxirane/onerun/r-methyloxirane_pbe_TZV2P-GTH_onerun-output-moments.dat", )
	# # uracil = ("/home/jmatti/projects/uracil/ehrenfest_gasphase_2/uracil_ehrenfest_gasphase_x-dir-output-moments.dat", "/home/jmatti/projects/uracil/ehrenfest_gasphase_2/uracil_ehrenfest_gasphase_y-dir-output-moments.dat", "/home/jmatti/projects/uracil/ehrenfest_gasphase_2/uracil_ehrenfest_gasphase_z-dir-output-moments.dat")
	# # files_p = ("/home/jmatti/projects/uracil/raman_all/results/uracil_mode_20_x-dir_p-output-moments.dat", "/home/jmatti/projects/uracil/raman_all/results/uracil_mode_20_y-dir_p-output-moments.dat", "/home/jmatti/projects/uracil/raman_all/results/uracil_mode_20_z-dir_p-output-moments.dat")
	# # files_m = ("/home/jmatti/projects/uracil/raman_all/results/uracil_mode_20_x-dir_m-output-moments.dat", "/home/jmatti/projects/uracil/raman_all/results/uracil_mode_20_y-dir_m-output-moments.dat", "/home/jmatti/projects/uracil/raman_all/results/uracil_mode_20_z-dir_m-output-moments.dat")


	# # Test Raman
	# files_ESGM_m = ("/home/jmatti/projects/methyloxirane/Raman_TZV2P-GTH_2/results/r-meth_mode_4_x-dir_m-output-moments.dat", "/home/jmatti/projects/methyloxirane/Raman_TZV2P-GTH_2/results/r-meth_mode_4_y-dir_m-output-moments.dat", "/home/jmatti/projects/methyloxirane/Raman_TZV2P-GTH_2/results/r-meth_mode_4_z-dir_m-output-moments.dat")
	# files_ESGM_p = ("/home/jmatti/projects/methyloxirane/Raman_TZV2P-GTH_2/results/r-meth_mode_4_x-dir_p-output-moments.dat", "/home/jmatti/projects/methyloxirane/Raman_TZV2P-GTH_2/results/r-meth_mode_4_y-dir_p-output-moments.dat", "/home/jmatti/projects/methyloxirane/Raman_TZV2P-GTH_2/results/r-meth_mode_4_z-dir_p-output-moments.dat")

	# test_Raman = Raman({'ts' : '0.1 au', 'files_p' : files_ESGM_p, 'files_m' : files_ESGM_m, 'code' : 'CP2K'})
	# # test_Raman.get_polarizabilities({'FT_method' : 'pade', 'pade_ws' : (0, 10, 0.01), 'start' : 0, 'stop' : 10000, 'broadening' : broadening})
	# # test_Raman.get_polarizabilities({'FT_method' : 'rfft', 'pade_ws' : (0, 10, 0.01), 'start' : 0, 'stop' : 100000, 'broadening' : broadening})
	# # test_Raman.get_polarizabilities({'FT_method' : 'pade', 'pade_ws' : 10, 'start' : 0, 'stop' : 10000, 'broadening' : broadening})
	# nu = 5.445591604207750152e+02
	# sQ = 7.331718720000000076e+02
	# excf = 3


	# test_Raman.get_polarizabilities({'FT_method' : 'pade', 'pade_ws' : (0, 10, 0.01), 'start' : 0, 'stop' : 40000, 'broadening' : broadening})
	# test_Raman.deriv(sQ, excf)
	# nu_p, sigma, f = test_Raman.intensity({'nu_p' : nu, 'T' : 300})
	# print(nu_p, sigma, f)

	# test_Raman.get_polarizabilities({'FT_method' : 'pade', 'pade_ws' : None, 'start' : 0, 'stop' : 40000, 'broadening' : broadening})
	# test_Raman.deriv(sQ, excf)
	# nu_p, sigma, f = test_Raman.intensity({'nu_p' : nu, 'T' : 300})
	# print(nu_p, sigma, f)

	# test_Raman.get_polarizabilities({'FT_method' : 'rfft', 'start' : 0, 'stop' : 100000, 'broadening' : broadening})
	# test_Raman.deriv(sQ, excf)
	# nu_p, sigma, f = test_Raman.intensity({'nu_p' : nu, 'T' : 300})
	# print(nu_p, sigma, f)

	# test_Raman.get_polarizabilities({'FT_method' : 'rfft',  'start' : 0, 'stop' : 100000, 'broadening' : None})
	# test_Raman.deriv(sQ, excf)
	# nu_p, sigma, f = test_Raman.intensity({'nu_p' : nu, 'T' : 300})
	# print(nu_p, sigma, f)

	# print(test_Raman.signal_damped.shape)
	# print([i[0] for i in test_Raman.ft_p])
	
	# print(type(test_Raman.ft_p))
	# print(test_Raman.deriv(1,2))
	# print(test_Raman.a_p)
	# plt.plot(np.arange(len(test_Raman.signal_damped[5,:])), test_Raman.signal_damped[5,:])
	# plt.plot(test_Raman.frequencies /const.e * const.h, np.imag(test_Raman.ft_p[5]))
	# plt.savefig('/home/jmatti/projects/output/art.pdf', format = 'pdf')
	# plt.show()

	print(debye2au(1))
	print(angstromtobohr(1))
	print(bohrtoangstrom(1))
	print(polarizability_autoSI(1))
	print(polarizability_deriv_autoSI(1))
	print(const.value('Bohr radius')/np.sqrt(const.m_e))