import re

"""
Rules for regular expressions:

- Prefix every regex with the quantum chemistry code name tag, e. g. CP2K, NWChem, TM
- store the values to be extracted in groups named as follows:
	- for single valued quantities: name the group "val"
	- for vectorial quantities: name the three groups "x", "y", and "z"
	- for tensorial quantities:	name the nine groups: 'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz'

"""


CP2K_EL_DIPOLE_REGEX=r"""
^\s* Dipole \s* moment\s*\[ \w* \] \s* 
	X= \s* (?P<x> [\-\+]?  ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? ) \s* 
	Y= \s* (?P<y> [\-\+]?  ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? ) \s* 
	Z= \s* (?P<z> [\-\+]?  ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? ) \s* 
	Total=\s* (?P<emom_tot> [\-\+]?  ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? ) \n 
"""

CP2K_MAG_DIPOLE_REGEX = r"""
^\s* Magnetic \s* Dipole \s* Moment \s* \[ \w* \] \s* 
	X= \s* (?P<x> [\-\+]?  ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? ) \s* 
	Y= \s* (?P<y> [\-\+]?  ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? ) \s* 
	Z= \s* (?P<z> [\-\+]?  ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? ) \s* 
	Total= \s* (?P<mmom_tot> [\-\+]?  ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? ) \n
"""


TM_EXCITATION_ENERGY_REGEX=r"""
^ [ \t]* Excitation\s* energy \s* \/ \s* eV: \s* (?P<val> [\-\+]?  ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? )
"""

TM_OSCSTRENGTH_REGEX=r"""
^ [ \t]* Oscillator \s* strength: \s*  \w* \s* \w*: \s* (?P<val> [\-\+]?  ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? ) 
"""

CP2K_EXCITATION_ENERGY_REGEX_NEW=r"""
 ^\s*TDDFPT\| \s*\d*\s* (?P<val> ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? ) .* \n 
"""

CP2K_OSCILLATOR_STRENGTH_REGEX_NEW=r"""
 ^\s*TDDFPT\| \s*\d*\s* ( [\-\+]? ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? \s* ) {4} (?P<val> [\*\+]? ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? ) .* \n 
"""

CP2K_EXCITATION_ENERGY_REGEX_OLD=r"""
 ^\s*excited\sstate\s: \s* \d* \s*  (?P<val> ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? ) .* \n 
"""
NWC_EL_DIPOLE_REGEX=r"""
 \s* (?P<x> [\-\+]?  ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? ) \s* 
 \s* (?P<y> [\-\+]?  ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? ) \s* 
 \s* (?P<z> [\-\+]?  ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? ) \s* 
"""

TM_RAMAN_NMF_REGEX = r"""
^\s*\d*\s*[a-z]*\s*(?P<val>( \d*\.\d+  | \d+\.?\d* )) 
"""

TM_RAMAN_INTENSITY_PERP = r"""
^\s*\d*\s*[a-z]*\s*(?P<nmf>( \d*\.\d+  | \d+\.?\d* )) \s* (YES | -) \s* [\-\+]? ( \d*\.\d+  | \d+\.?\d* ) \s* [\-\+]? ( \d*\.\d+  | \d+\.?\d* ) \s* (?P<val> ( [\-\+]? ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? \s* ) )  \s* ( ( [\-\+]? ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? \s* ) ) \n
"""

TM_RAMAN_INTENSITY_PAR = r"""
^\s*\d*\s*[a-z]*\s*(( \d*\.\d+  | \d+\.?\d* )) \s* (YES | -) \s* [\-\+]? ( \d*\.\d+  | \d+\.?\d* ) \s* [\-\+]? ( \d*\.\d+  | \d+\.?\d* ) \s* ( ( [\-\+]? ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? \s* ) )  \s* (?P<val> ( [\-\+]? ( \d*\.\d+  | \d+\.?\d* )  ([Ee][\+\-]?\d+)? \s* ) ) \n
"""

# compile regular expressions

exc_match = re.compile(TM_EXCITATION_ENERGY_REGEX.encode('utf-8'), re.MULTILINE | re.VERBOSE)
osc_match = re.compile(TM_OSCSTRENGTH_REGEX.encode('utf-8'), re.MULTILINE | re.VERBOSE)
cp2k_edipole_match = re.compile(CP2K_EL_DIPOLE_REGEX.encode('utf-8'), re.MULTILINE | re.VERBOSE)
cp2k_mdipole_match = re.compile(CP2K_MAG_DIPOLE_REGEX.encode('utf-8'), re.MULTILINE | re.VERBOSE)
cp2k_exc_ener_new_match = re.compile(CP2K_EXCITATION_ENERGY_REGEX_NEW.encode('utf-8'), re.MULTILINE | re.VERBOSE)
cp2k_osc_str_new_match = re.compile(CP2K_OSCILLATOR_STRENGTH_REGEX_NEW.encode('utf-8'), re.MULTILINE | re.VERBOSE)
cp2k_exc_ener_old_match = re.compile(CP2K_EXCITATION_ENERGY_REGEX_OLD.encode('utf-8'), re.MULTILINE | re.VERBOSE)
nwchem_edipole_match = re.compile(NWC_EL_DIPOLE_REGEX.encode('utf-8'), re.MULTILINE | re.VERBOSE)
tm_raman_nmf_match = re.compile(TM_RAMAN_NMF_REGEX.encode('utf-8'), re.MULTILINE | re.VERBOSE)
tm_raman_int_perp_match = re.compile(TM_RAMAN_INTENSITY_PERP.encode('utf-8'), re.MULTILINE | re.VERBOSE)
tm_raman_int_par_match = re.compile(TM_RAMAN_INTENSITY_PAR.encode('utf-8'), re.MULTILINE | re.VERBOSE)
# cp2k_mdipole_berry =
# exc_match = re.compile(TM_EXCITATION_ENERGY_REGEX, re.MULTILINE | re.VERBOSE)
# osc_match = re.compile(TM_OSCSTRENGTH_REGEX, re.MULTILINE | re.VERBOSE)
# cp2k_edipole_match = re.compile(CP2K_EL_DIPOLE_REGEX, re.MULTILINE | re.VERBOSE)
# cp2k_mdipole_match = re.compile(CP2K_MAG_DIPOLE_REGEX, re.MULTILINE | re.VERBOSE)


# create dictionary with tags for choosing the right regular expression

CP2K_tags = {"RT-edipole" : cp2k_edipole_match, "RT-mdipole" : cp2k_mdipole_match, "TDDFPT_exc_ener_new" : cp2k_exc_ener_new_match, "TDDFPT_osc_str_new" : cp2k_osc_str_new_match, "TDDFPT_exc_ener_old" : cp2k_exc_ener_old_match}
TM_tags = {"exc_ener" : exc_match, "osc_str" : osc_match, "Raman-nmf" : tm_raman_nmf_match, "Raman-int-perp" : tm_raman_int_perp_match, "Raman-int-par" : tm_raman_int_par_match}
NWC_tags = {"RT-edipole" : nwchem_edipole_match}

tagdict = {"CP2K" : CP2K_tags, "TM" : TM_tags, "NWC" : NWC_tags }

dimdict = {"RT-edipole" : 'vec', "RT-mdipole" : 'vec', "exc_ener" : 'val', "osc_str" : 'val', "TDDFPT_exc_ener_new" : 'val', "TDDFPT_osc_str_new" : 'val', "TDDFPT_exc_ener_old" : 'val', "Raman-nmf" : 'val', "Raman-int-perp" : 'val', "Raman-int-par" : 'val'}


# function to decompose the match iterators into groups

def degroup(prop, matchiter):
	if dimdict[prop] == 'val':
		for i in matchiter:
			yield float(i.group('val'))

	elif dimdict[prop] == 'vec':
		for i in matchiter:
			yield (float(i.group('x')), float(i.group('y')), float(i.group('z')))

	elif dimdict[prop] == 'ten':
		for i in matchiter:
			yield (float(i.group('xx')), float(i.group('xy')), float(i.group('xz')), float(i.group('yx')), float(i.group('yy')), float(i.group('yz')), float(i.group('zx')), float(i.group('zy')), float(i.group('zz')))

	else:
		raise KeyError("unknown property format")

def fileformat(prop, data):
	if dimdict[prop] == 'val':
		return "{}\n".format(data)

	elif dimdict[prop] == 'vec':
		return "{}\t{}\t{}\n".format(data[0], data[1], data[2])

	elif dimdict[prop] == 'ten':
		return "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8])

	else:
		raise KeyError("unknown property format")

def writelegend(prop):
	if dimdict[prop] == 'val':
		return "{}\n".format('Value')

	elif dimdict[prop] == 'vec':
		return "{}\t{}\t{}\n".format('x','y','z')

	elif dimdict[prop] == 'ten':
		return "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format('xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz')

	else:
		raise KeyError("unknown property format")