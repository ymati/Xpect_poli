from Xpect_new import regex
import os
import numpy as np
import mmap
import contextlib

class Xtract:
	""" 
	Extract data from quantum chemistry package output files using regular expressions specified in regex.py

	Input: 	path to data file
	       	Nametag (hard-coded) of the quantum chemistry code
	       	List of nametags (hard-coded) for properties to be extracted from the file

	Methods:	parse():		returns generators containing the values for each property
				extract():		returns a numpy array containg the requested properties
				write2file():	writes the extracted data to a file for each property


	Regular expressions, nametags and formatting functions are in regex.py

	"""

	def __init__(self, path, code = 'CP2K', properties = ('RT-edipole',)):		
		self.code = code
		self.props = properties

		# check if the nametags for code and properties are specified in regex.py
		if self.code not in regex.tagdict:
			print('Tag:', self.code)
			raise KeyError('Requested quantum chemistry package name tag spelled wrong or not available (yet)')

		if all(prop not in regex.tagdict[self.code] for prop in self.props):
			raise KeyError('Requested property name tag spelled wrong or no matching regular expression found')
		
		self.path = os.path.abspath(path)
		# print(self.path)



	def _parse(self):
		"""
		Matches the regular expression for a property and returns a generator containing a generator yielding the requested data for each property
		"""
		with open(self.path, 'r') as f:
			with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as content:
				for prop in self.props:
					yield regex.degroup(prop, regex.tagdict[self.code][prop].finditer(content))

	# def parse(self):
	# 	with open(self.path, 'r') as f:
	# 		content = f.read()
	# 		for prop in self.props:
	# 			yield regex.degroup(prop, regex.tagdict[self.code][prop].finditer(content))
			

	def extract(self):
		"""
		Reads out the generator from the parse function and returns a numpy array for further manipulations: [array(property 0), array(propertiy 1), ...]
		"""
		return np.asarray(tuple(list(i) for i in self._parse())) 
		


	def write2file(self, projname = None):
		"""
		Reads out the generator from the parse function and writes requested propierties to data-files: "$projname_extracted_$property.dat" in the same directory as the original file
		"""
		projname = projname if projname is not None else os.path.splitext(os.path.split(self.path)[1])[0] + '_extracted'

		# create filenames and headers
		fnames = []
		headers = []
		for i in self.props:
			fnames.append("{}{}{}{}{}{}".format(os.path.split(self.path)[0],"/", projname, "_", i ,'.dat'))
			headers.append("{}{}\t{}{}\n".format("# ", i, "from: ", self.path))

		
		# read out the generators and write to file using the formating functions in regex.py
		count = 0
		for i in self._parse():			
			data = ''
			for j in list(i):
				data += regex.fileformat(self.props[count], j)

			with open(fnames[count], 'w') as fout:
				fout.write(headers[count] + data)

			count += 1


if __name__ == "__main__":

	import argparse

	parser = argparse.ArgumentParser(description='extract data from output files of quantum chemistry codes')
	parser.add_argument("filename", help="filename of qc-code output file")
	parser.add_argument("code", help="Name tag of the qc-code, e. g. CP2K, TM, NWC")
	parser.add_argument("props", help="properties to be extracted, e. g. RT-edipole, RT-mdipole, osc_str separated by ', ' as string", type = str)
	parser.add_argument("-pn", dest='projectname', help="optional projectname for the naming of output-files", default = None)
	args = parser.parse_args()

	test = Xtract(args.filename, code = str(args.code), properties = tuple(args.props.split(", ")))
	test.write2file(str(args.projectname))


	# p, m = test.extract()
	# print(p,m)
