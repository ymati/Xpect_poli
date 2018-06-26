from Xpect_new import Xtract

# return (excitation_energy, oscillator_strength) independent of the quantum chemistry code
def standardize_data(args):
	if args['code'] == 'TM':
		print(args['files'][0])
		print(args['properties'])
		data = Xtract.Xtract(path = args['files'][0], code = 'TM', properties = args['properties']).extract()

		return {i : data[index] for index, i in enumerate(args['properties'])}

	if args['code'] == 'CP2K':
		if args['version'] == 'new':
			args['properties'] = ('TDDFPT_exc_ener_new', 'TDDFPT_osc_str_new')
			return Xtract.Xtract(path = args['files'][0], code = 'CP2K', properties = args['properties']).extract()
		if args['version'] == 'old':
			args['properties'] = ('TDDFPT_exc_ener_old', )
			return Xtract.Xtract(path = args['files'][0], code = 'CP2K', properties = args['properties']).extract()

	else:
		raise KeyError("Quantum chemistry code not implemented with LR-TDDFT yet")


class absorption:
	def __init__(self, args):
		args['properties'] = ('exc_ener', 'osc_str')
		self.data = standardize_data(args)


	def spectrum(self, args):
		return self.data['exc_ener'], self.data['osc_str']

class Raman:
	def __init__(self, args):
		args['properties'] = ('Raman-nmf', 'Raman-int-perp', 'Raman-int-par')
		self.data = standardize_data(args)

	def spectrum_perp(self):
		return self.data['Raman-nmf'], self.data['Raman-int-perp']

	def spectrum_par(self):
		return self.data['Raman-nmf'], self.data['Raman-int-par']