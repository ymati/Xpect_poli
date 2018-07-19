import numpy as np
from scipy import constants as const 


class absorption_RT:
	def __init__(self, method = 'RT', **kwargs):
		from Xpect_new import RT as meth
		self.spec = meth.absorption(kwargs)

	def get_polarizabilities(self, **kwargs):
		return self.spec.get_polarizabilities(kwargs)

	def spectrum(self, **kwargs):
		return self.spec.spectrum(kwargs)

	def _scan(self, f):
		return self.spec._scan(f)

	def exc_ener(self, lower, upper):
		return self.spec.exc_ener(lower, upper)

	def mean_polarizability(self, w, kappa):
		return self.spec.mean_polarizability(w, kappa)


class FT:
	def __init__(self, method = 'RT', **kwargs):
		from Xpect_new import RT as meth
		self.spec = meth.FT(kwargs)

	def get_polarizabilities(self, **kwargs):
		return self.spec.get_polarizabilities(kwargs)

	def FT(self, **kwargs):
		return self.spec.FT(kwargs)

	def dipole_moment(self):
		return self.spec.signal


class Raman_RT:
	"""
	Real time Raman spectra
	input:
	for each normal mode:
		- files_p, files_m:	2 x 3 RTP dipole signals for the polarizability tensor (Fourier transform), displaced in positive and negative direction along the normal mode as tuples of three
		- ts: 				time step used in the calculations 
		- code:				Quantum chemistry code name tag for the RTP code used e. g. 'CP2K'

	methods:
		- get_polarizabilities: calculate polarizability from the dipole signals
			options:
			- FT_method: 		Method used for the Fourier transform, either 'rfft' (Real fast Fourier transform) or 'pade' (Padé approximants).
			- broadening: 		Damping factor for the time domain signals in a. u. e. g. 0.0037
			- start, stop:		Integer parameters to slice the part of the time domain signals which are used for the Fourier transform
			- pade_ws:			Define frequency range and stepsize for Padé approximants
								leave empty (None) if only the polarizability at a certain frequency is of interest and not the whole spectrum.
		- deriv:				takes the numerical derivative of the polarizabilities required for the RT-Raman spectrum
								diff is the finite difference in the denominator
								freq is the (excitation) frequency at which the derivative is taken
		- intensity:			input:
									- nu_p			normal mode frequency [cm^-1]
									- exc_freq		excitation frequency [eV]
									- T				Temperature [K]
								output:	(nu_p, d_sigma, f_exc)	
									- nu_p			normal mode frequency [cm^-1]
									- d_sigma		Raman differential cross section
									- f_exc			Excitation frequency [eV] 
	"""							
	def __init__(self, method = 'RT', **kwargs):
		self.method = method
		if self.method == 'RT':
			from Xpect_new import RT as meth

		else:
			raise KeyError("specified method unkown")

		self.spec = meth.Raman(kwargs)

	def get_polarizabilities(self, **kwargs):
		return self.spec.get_polarizabilities(kwargs)

	def deriv(self, diff, freq):
		return self.spec.deriv(diff, freq)

	def intensity(self, **kwargs):
		return self.spec.intensity(kwargs)


class ESGM_RT:
	"""
	Real time excited state gradient method
	input:
	for each normal mode:
		- files_p, files_m:	2 x 3 RTP dipole signals for xx, yy and zz component of the polarizability (Fourier transform), displaced in positive and negative direction along the normal mode as tuples of three
		- ts: 				time step used in the calculations 
		- code:				Quantum chemistry code name tag for the RTP code used e. g. 'CP2K'

	methods:
		- get_polarizabilities: calculate polarizability from the dipole signals
			options:
			- FT_method: 		Method used for the Fourier transform, either 'rfft' (Real fast Fourier transform) or 'pade' (Padé approximants). For the ESGM_RT it necessarily needs to be 'pade' 
			- broadening: 		Damping factor for the time domain signals in a. u. e. g. 0.0037
			- start, stop:		Integer parameters to slice the part of the time domain signals which are used for the Fourier transform
			- pade_ws:			Define frequency range and stepsize for Padé approximants 
								leave empty (None) if only the polarizability at a certain frequency is of interest and not the whole spectrum, i. e. leave empty for ESGM_RT
		- scan_p/m:				Return the absoprtion cross section for either the positively (p) or negatively (m) displaced geometry at the specified frequency w [eV]
		- exc_ener_p/m 			Searches for an absorption maximum between $lower [eV] and $upper [eV] using the golden section search algorithm and the scan_p/m methods as functions
		- gradient 				Returns (E^ex_p - E^ex_m)/fd, i. e. the numerical gradient of the excitation energy w. r. t. the normal mode; lower [eV] and upper [eV] the range where the excitation energy is to be found from the spectrum

	"""
	def __init__(self, method = 'RT', **kwargs):
		self.method = method
		if self.method == 'RT':
			from Xpect_new import RT as meth

		else:
			raise KeyError("specified method unkown")

		self.spec = meth.ESGM_2(kwargs)

	def get_polarizabilities(self, **kwargs):
		return self.spec.get_polarizabilities(kwargs)

	def scan_p(self, w):
		return self.spec.scan_p(w)
	
	def scan_m(self, w):
		return self.spec.scan_m(w)

	def exc_ener_p(self, lower, upper):
		return self.spec.exc_ener_p(lower, upper)

	def exc_ener_m(self, lower, upper):
		return self.spec.exc_ener_m(lower, upper)

	def gradient(self, lower, upper, fd):
		return self.spec.gradient(lower, upper, fd)

class Raman_LR:
	def __init__(self, method = 'LR', **kwargs):
		self.method = method

		if self.method == 'LR':
			from Xpect_new import LR as meth

		else:
			raise KeyError("specified method unkown")

		self.spec = meth.Raman(kwargs)

	def spectrum_perp(self):
		return self.spec.spectrum_perp()

	def spectrum_par(self):
		return self.spec.spectrum_par()


class ECD:
	def __init__(self, method = 'RT', **kwargs):
		if method == 'RT':
			from Xpect_new import RT as meth

		elif method == "LR":
			from Xpect_new import LR as meth

		else:
			raise KeyError("specified method unkown")

		self.spec = meth.ECD(kwargs)

	def spectrum(self, **kwargs):
		print("ECD!!!")
		return self.spec.spectrum(kwargs)


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	broadening = 0.1 * const.e / const.value('Hartree energy') #
	print(broadening)

	files = ("/home/jmatti/projects/uracil/functionals_ts_au/functionals/uracil_pbe_rtp_x-dir-output-moments.dat", "/home/jmatti/projects/uracil/functionals_ts_au/functionals/uracil_pbe_rtp_y-dir-output-moments.dat", "/home/jmatti/projects/uracil/functionals_ts_au/functionals/uracil_pbe_rtp_z-dir-output-moments.dat",)
	files_lr = ("/home/jmatti/projects/uracil/tm_lrtddft/pbe/escf.out", )
	files_mag = ("/home/jmatti/projects/methyloxirane/ts_au_TZVP/r-methyloxirane_pbe_x-dir_TZVP-output-moments.dat", "/home/jmatti/projects/methyloxirane/ts_au_TZVP/r-methyloxirane_pbe_y-dir_TZVP-output-moments.dat", "/home/jmatti/projects/methyloxirane/ts_au_TZVP/r-methyloxirane_pbe_z-dir_TZVP-output-moments.dat")
	files_mag_s = ("/home/jmatti/projects/methyloxirane/ts_au_TZVP/s-methyloxirane_pbe_x-dir_TZVP-output-moments.dat", "/home/jmatti/projects/methyloxirane/ts_au_TZVP/s-methyloxirane_pbe_y-dir_TZVP-output-moments.dat", "/home/jmatti/projects/methyloxirane/ts_au_TZVP/s-methyloxirane_pbe_z-dir_TZVP-output-moments.dat")
	files_p = ("/home/jmatti/projects/uracil/raman_all/results/uracil_mode_19_x-dir_p-output-moments.dat", "/home/jmatti/projects/uracil/raman_all/results/uracil_mode_19_y-dir_p-output-moments.dat", "/home/jmatti/projects/uracil/raman_all/results/uracil_mode_19_z-dir_p-output-moments.dat")
	files_m = ("/home/jmatti/projects/uracil/raman_all/results/uracil_mode_19_x-dir_m-output-moments.dat", "/home/jmatti/projects/uracil/raman_all/results/uracil_mode_19_y-dir_m-output-moments.dat", "/home/jmatti/projects/uracil/raman_all/results/uracil_mode_19_z-dir_m-output-moments.dat")
	files_conv = ("/home/jmatti/projects/methyloxirane/basis_sets_pbe/r-methyloxirane_pbe_QZV2P-GTH_E-12_x-dir-output-moments.dat","/home/jmatti/projects/methyloxirane/basis_sets_pbe/r-methyloxirane_pbe_QZV2P-GTH_E-12_y-dir-output-moments.dat","/home/jmatti/projects/methyloxirane/basis_sets_pbe/r-methyloxirane_pbe_QZV2P-GTH_E-12_z-dir-output-moments.dat")


	# uracil_lr = absorption(method = 'LR', codefiles = {'code' : 'TM', 'files' : files_lr})
	# r_methyloxirane = ECD(method = 'RT', ts = '0.5 au', code = 'CP2K', files = files_mag)
	# s_methyloxirane = ECD(method = 'RT', ts = '0.5 au', codefiles = {'code' : 'CP2K', 'files' : files_mag_s})
	uracil = absorption(method = 'RT', ts = '0.5 au', code = "CP2K", files = files ) 
	methyloxirane_conv = absorption(method = 'RT', ts = '0.5 au', code = 'CP2K', files = files_conv)

	x_pade, y_pade = uracil.spectrum(broadening = None)
	x, y = methyloxirane_conv.spectrum() # broadening = broadening, FT_method = 'pade', pade_ws = (0, 20, 0.001), start = 0, stop = 10000)
	# x_br, y_br = uracil.spectrum(broadening = 0.1 * const.e * const.value('Hartree energy') / const.hbar)
	x_2, y_2 = methyloxirane_conv.spectrum(broadening = None, start = 0, stop = 20000)
	# x_ecd, y_ecd = r_methyloxirane.spectrum() #FT_method = 'pade', pade_ws = (0, 20, 0.001), start = 0, stop = 10000, broadening = broadening)
	# x_ecd_s, y_ecd_s = s_methyloxirane.spectrum(FT_method = 'pade', pade_ws = (0, 20, 0.0001), start = 0, stop = 20000, broadening = broadening)


# Raman testing
	# Raman_test = Raman_RT(ts = '0.5 au', code = 'CP2K', files_p = files_p, files_m = files_m)

	# Raman_test.get_polarizabilities(broadening = broadening, FT_method = 'pade', pade_ws = (0, 20, 0.001), start = 0, stop = 10000)
	# Raman_test.deriv(0.0236539388553)
	# print(Raman_test.intensity(exc_freq = 4.71, nu_p = 1193.010501, T = 300))

	# Raman_test.get_polarizabilities(FT_method = 'pade', pade_ws = (0, 20, 0.001), start = 0, stop = 10000, broadening = broadening)
	# Raman_test.deriv(1000)
	# print(Raman_test.intensity(exc_freq = 4.71, nu_p = 1193.010501, T = 300))


	# Raman_test.get_polarizabilities(broadening = broadening, FT_method = 'pade', pade_ws = (0, 20, 0.001), start = 0, stop = 10000)
	# Raman_test.deriv(0.0236539388553)
	# print(Raman_test.intensity(exc_freq = 3.2, nu_p = 1193.010501, T = 300))
	# # print(x_lr, y_lr)

	# Raman_test.get_polarizabilities(broadening = broadening, FT_method = 'pade', pade_ws = (0, 20, 0.001), start = 0, stop = 10000)
	# Raman_test.deriv(1000)
	# print(Raman_test.intensity(exc_freq = 3.2, nu_p = 1193.010501, T = 300))


# plotting
	fig = plt.figure(1)

	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)

	ax1.plot(x*const.h/const.e, y, label = 'absorption - None ')
	ax1.plot(x_2*const.h/const.e, y_2, label = 'absorption - broadened')
	ax1.set_xlim(0, 20)
	# ax1.set_ylabel('absorption')
	# ax1.set_ylim(0.99e9, 1.1e9)
	# # ax1.set_ylim(0.96e9,1e9)
	ax1.grid()
	ax2.plot(x_pade*const.h/const.e, y_pade, label = 'uracil - broadened')
	# # ax2.plot(x_ecd*const.h/const.e, y_ecd)
	# # ax2.plot(x_ecd_s*const.h/const.e, y_ecd_s)
	# ax2.set_ylabel('absorption')
	ax2.set_xlim(0, 20)
	# ax2.set_ylim(-1e7, 0.1e9)
	ax2.grid()
	ax1.legend(loc = 'upper right', prop = {'size' : 8})
	# ax2.legend(loc = 'upper right', prop = {'size' : 8})
	# plt.xlabel('excitation energy [eV]')
	# # plt.savefig('/home/jmatti/ownCloud/Documents/Progress report 3/uracil_pade_broad_.pdf', format='pdf')

	fig.tight_layout()
	plt.show()

