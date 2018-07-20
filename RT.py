from Xpect_new import Xtract
from Xpect_new import convert_units as conv
import numpy as np 
from scipy import constants as const
import re
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

def standardize_signals(args):

	if args['code'] not in ["CP2K", "NWC"]:
		raise KeyError("Quantum chemistry code not implemented with RT-TDDFT yet")
	print('reading data files...')

	sign = [Xtract.Xtract(path = i, code = args['code'], properties = args['properties'] ).extract() for i in args['files']]
	# axes are [file, property, timestep, values (i.e. value or vector)]

	min_length = len(sign[0][0])
	print(min_length)

	# cut signals to have the same ammount of timesteps
	if any(len(j) != min_length for i in sign for j in i):
		
		min_length = min([len(j) for i in sign for j in i])
		print('The lengths of the different dipole signals do not match and will be adapted to the shortest: ', min_length)

		sign_short = np.empty([len(sign), len(args['properties']), min_length, 3])
		for i, prop in enumerate(sign):
			for j, signal in enumerate(prop):
				for k in range(min_length):
					sign_short[i,j,k] = signal[k]


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

	@staticmethod
	def FT_unified(signal, timestep, kwargs):
		ft_length = kwargs['stop'] - kwargs['start']
		if kwargs['FT_method'] == 'rfft':
			print('rfft ...')
			single_point = False

			frequencies = np.fft.rfftfreq(ft_length, timestep)*const.value('Hartree energy')/const.hbar
			ft = [-np.fft.rfft(sig) for sig in signal]

		# Pade approximation
		elif kwargs['FT_method'] == 'pade':
			print('Pade approximants...')
			if isinstance(kwargs['pade_ws'], tuple):
				single_point = False
				w_start = conv.ev2au(kwargs['pade_ws'][0])
				w_stop = conv.ev2au(kwargs['pade_ws'][1])
				w_step = conv.ev2au(kwargs['pade_ws'][2])
				freq_pade = np.arange(w_start, w_stop, w_step)
				frequencies = freq_pade *const.value('Hartree energy')/const.hbar
				ft = tuple([pade(sig, ft_length, timestep, freq_pade, single_point = single_point) for sig in signal])
				# print('tuple')

			else:
				single_point = True
				frequencies = []
				ft = tuple([pade(sig, ft_length, timestep, 0, single_point = single_point) for sig in signal])

		else:
			raise KeyError("FT_method not known or misspelled. Currently: rfft, pade")

		return frequencies, ft, single_point

	@staticmethod
	def damping_signal(signal_read, timestep, kwargs):
		ft_length = kwargs['stop'] - kwargs['start']
		if kwargs['broadening'] is not None:
			print('damping signal ...')
			factor = np.exp(-(timestep*np.arange(ft_length)*float(kwargs['broadening'])))*timestep 						# * timestep already for the FT
			signal = np.asarray([conv.debye2au(sig)*factor for sig in signal_read[:,kwargs['start']:kwargs['stop']]])
		else:
			signal = debye2au(signal_read[:,kwargs['start']:kwargs['stop']])
		return signal

###################################################################################################
#####				Peform Fourier transfrom of the el. dipole signal 						  #####
###################################################################################################

class FT():
	def __init__(self, kwargs):
		kwargs_read = {'ts' : kwargs['ts'], 'code' : kwargs['code'], 'files' : kwargs['files'], 'diag' : False, 'properties' : ('RT-edipole', ) } 

		self.read = randft(kwargs_read)

		self.timestep = self.read.timestep
		self.signal_length = self.read.signal_length

	def get_polarizabilities(self, kwargs):
		ft_args = parse_ft_args(kwargs, self.signal_length)

		# damp signal
		self.signal = self.read.damping_signal(self.read.signals['RT-edipole'], self.timestep, ft_args)

		# do FT either with FFT or Pade
		self.frequencies, self.ft, self.sp = self.read.FT_unified(self.signal, self.timestep, ft_args)


	def FT(self, kwargs):
		return self.frequencies, np.real(self.ft), np.imag(self.ft) #, self.signal_damped


###################################################################################################
#####				Absorption spectrum and some usuful tools							      #####
###################################################################################################

class absorption:
	def __init__(self, kwargs):
		kwargs_read = {'ts' : kwargs['ts'], 'code' : kwargs['code'], 'files' : kwargs['files'], 'diag' : True, 'properties' : ('RT-edipole', ) } 

		self.read = randft(kwargs_read)

		self.timestep = self.read.timestep
		self.signal_length = self.read.signal_length

	def get_polarizabilities(self, kwargs):
		ft_args = parse_ft_args(kwargs, self.signal_length)

		# damp signal
		self.signal = self.read.damping_signal(self.read.signals['RT-edipole'], self.timestep, ft_args)

		# do FT either with FFT or Pade
		self.frequencies, self.ft, self.sp = self.read.FT_unified(self.signal, self.timestep, ft_args)


	def spectrum(self, kwargs):
		kappa = kwargs['kappa'] if 'kappa' in kwargs else float(1)
		trace = np.asarray(self.ft).reshape((3, len(self.frequencies)))
		return conv.f2eV(self.frequencies), 4 * np.pi * self.frequencies / (3 * const.c * kappa) * np.imag(trace[0] + trace[1] + trace[2])

	def _scan(self, w):
		W = np.exp(-1j * conv.ev2au(w) * self.timestep * 2 * np.pi)

		trace = [i(W)/j(W) for (i, j) in self.ft]

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

		self.timestep = self.read_p.timestep
		self.signal_length = min(self.read_p.signal_length, self.read_m.signal_length)

	def get_polarizabilities(self, kwargs):
		ft_args = parse_ft_args(kwargs, self.signal_length)

		if 'fieldstrength' in kwargs:
			self.fieldstrength = kwargs['fieldstrength']
		else:
			self.fieldstrength = 1
			print('fieldstrength was set to 1 [au]')

		# damp signal
		signal_p = self.read_p.damping_signal(self.read_p.signals['RT-edipole'], self.timestep, ft_args)
		signal_m = self.read_m.damping_signal(self.read_m.signals['RT-edipole'], self.timestep, ft_args)

		# do FT 
		self.frequencies, self.ft_p, self.single_point = self.read_p.FT_unified(signal_p, self.timestep, ft_args)
		frequencies_m, self.ft_m, sp = self.read_m.FT_unified(signal_m, self.timestep, ft_args)
		

	def deriv(self, diff, freq): 		# take derivative at a specific excitation frequency
		if self.single_point:			
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
				


		# print('polarizability_p: ', self.a_p)
		# print('polarizability_m: ', self.a_m)
		# print('polarizability_m_SI: ', conv.polarizability_au2SI(self.a_m))
		# print('mean polarizability_p: ', 1/3*(self.a_p[0,0] + self.a_p[1,1] + self.a_p[2,2]))
		# print('mean polarizability_m: ', 1/3*(self.a_m[0,0] + self.a_m[1,1] + self.a_m[2,2]))
		# print('difference: ', self.a_p - self.a_m)
		# print('diff', diff)
		self.da = diff_3(self.a_p, self.a_m, diff)

		self.da_SI = conv.polarizability_deriv_au2SI(self.da / np.sqrt(const.m_e/const.value('atomic mass constant')) ) / np.sqrt(const.m_e)  # convert to SI units and account for mass-weighted coordinates

		print(self.da)


	def intensity(self, kwargs):
		nu_p = kwargs['nu_p']

		# ak = 1/float(3) * np.abs( self.da[0,0] + self.da[1,1] + self.da[2,2])

		# gamma_k2 = 1/float(2) * ( np.abs(self.da[0,0] - self.da[1,1])**2 + np.abs(self.da[1,1] - self.da[2,2])**2 + np.abs(self.da[2,2] - self.da[0,0])**2  + 6 * ( np.abs(self.da[0,1])**2 + np.abs(self.da[1,2])**2 + np.abs(self.da[2,0])**2 ) )

		# print('(45 a_k**2 + 7 gamma_k**2) (angstrom**4/amu ?): \t', (45*ak**2 + 7 * gamma_k2)*conv.bohr2angstrom(1)**4)

		ak_SI = 1/float(3) * np.abs( self.da_SI[0,0] + self.da_SI[1,1] + self.da_SI[2,2])

		gamma_k2_SI = 1/float(2) * ( np.abs(self.da_SI[0,0] - self.da_SI[1,1])**2 + np.abs(self.da_SI[1,1] - self.da_SI[2,2])**2 + np.abs(self.da_SI[2,2] - self.da_SI[0,0])**2  + 6 * ( np.abs(self.da_SI[0,1])**2 + np.abs(self.da_SI[1,2])**2 + np.abs(self.da_SI[2,0])**2 ) )

		d_sigma = np.pi**2/(const.epsilon_0**2) * (conv.f2nu_SI(self.resonancefreq) - nu_p*100)**4 * const.h / (8 * np.pi**2 * const.c * nu_p*100) * (45*ak_SI**2 + 7 * gamma_k2_SI)/45 * 1 / (1 - np.exp(- const.h * const.c * nu_p*100 / ( const.k * kwargs['T'] )))

		self.frequency_factor = np.pi**2/(const.epsilon_0**2) * (conv.f2nu_SI(self.resonancefreq) - nu_p*100)**4 * const.h / (8 * np.pi**2 * const.c * nu_p*100)
		self.Raman_activity_SI = (45*ak_SI**2 + 7 * gamma_k2_SI)/45
		self.temperature_factor =  1 / (1 - np.exp(- const.h * const.c * nu_p*100 / ( const.k * kwargs['T'] )))

		print('frequency factor: \t\t', np.pi**2/(const.epsilon_0**2) * (conv.f2nu_SI(self.resonancefreq) - nu_p*100)**4 * const.h / (8 * np.pi**2 * const.c * nu_p*100))
		print('(45 a_k**2 + 7 gamma_k**2)/45 (SI): \t', (45*ak_SI**2 + 7 * gamma_k2_SI)/45)
		print('Temperature factor: \t\t', 1 / (1 - np.exp(- const.h * const.c * nu_p*100 / ( const.k * kwargs['T'] ))))

		return nu_p, d_sigma, conv.f2eV(self.resonancefreq)

###################################################################################################
#####		RT-TDDFT Excited state gradient method										      #####
###################################################################################################

class ESGM:
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
		ft_args = parse_ft_args(kwargs, self.signal_length)

		if 'fieldstrength' in kwargs:
			self.fieldstrength = kwargs['fieldstrength']
		else:
			self.fieldstrength = 1
			print('fieldstrength was set to 1 [au]')

		# damping the signal
		signal_p = self.read_p.damping_signal(self.read_p.signals['RT-edipole'], self.timestep, ft_args)
		signal_m = self.read_m.damping_signal(self.read_m.signals['RT-edipole'], self.timestep, ft_args)

		# Pade approximation
		frequencies_p, self.ft_p, sp = self.read_p.FT_unified(signal_p, self.timestep, ft_args)
		frequencies_m, self.ft_m, sp = self.read_m.FT_unified(signal_m, self.timestep, ft_args)


	def scan_p(self, w):
		W = np.exp(-1j * conv.ev2au(w) * self.timestep * 2 * np.pi)
		trace = [i(W)/j(W)/self.fieldstrength for (i, j) in self.ft_p]

		f = conv.eV2f(w) 				# convert to Hz [SI]
		return 4 * np.pi * f / (3 * const.c) * np.imag(trace[0] + trace[1] + trace[2])		
	
	def scan_m(self, w):
		W = np.exp(-1j * conv.ev2au(w) * self.timestep * 2 * np.pi)
		trace = [i(W)/j(W)/self.fieldstrength for (i, j) in self.ft_m]

		f = conv.eV2f(w) 				# convert to Hz [SI]
		return 4 * np.pi * f / (3 * const.c) * np.imag(trace[0] + trace[1] + trace[2])		

	def exc_ener_p(self, lower, upper):
		return gss(self.scan_p, lower, upper)

	def exc_ener_m(self, lower, upper):
		return gss(self.scan_m, lower, upper)

	def gradient(self, lower, upper, fd):
		return np.abs(self.exc_ener_p(lower, upper) - self.exc_ener_m(lower, upper))/fd

if __name__ == "__main__":
	from matplotlib import pyplot as plt