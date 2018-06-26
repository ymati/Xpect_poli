from scipy import constants as const
import numpy as np

def eV2f(x):
	return x /const.h * const.e

def f2eV(x):
	return x * const.h/const.e

def f2nu(x):
	return x / (const.c*100)

def f2nu_SI(x):
	return x / const.c

def ev2au(x):
	return x * const.e  / const.value('Hartree energy') / (2*np.pi)

def f2eV(x):
	return x*const.h/const.e

def debye2au(x):
	return 1/const.c / (const.value('Bohr radius')*const.e) *x*1e-21
#	return const.c * (const.value('Bohr radius')*const.e) *x/1e-21

def angstrom2bohr(x):
	return 1/(const.value('Bohr radius') *1e10) * x

def bohr2angstrom(x):
	return const.value('Bohr radius') * 1e10 * x

def polarizability_au2SI(x):
	return const.e**2 * const.value('Bohr radius')**2 / const.value('Hartree energy') * x

def polarizability_deriv_au2SI(x):
	# return 1/(const.e**2 * const.value('Bohr radius')) * const.value('Hartree energy') * x
	return const.e**2 * const.value('Bohr radius') / const.value('Hartree energy') * x

def fs2au(x):
	return const.value('Hartree energy')/const.hbar * x * 1e-15 