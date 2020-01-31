import warnings
import numpy as np

class InputImage(object):
	def __init__(self, image_array, height_in_pix, bidirectional_enabled):
		self.bidirectional_enabled = bidirectional_enabled
		self.image_array = image_array
		self.height_in_pix = height_in_pix
		if len(self.image_array) % self.height_in_pix != 0:
			raise Exception("Incorrect number of pixels in input image")

		self.image_array = self.image_array.reshape(-1, self.height_in_pix)
		self.image_array = self.image_array.T

		if self.image_array.shape[1] % 2 != 0:
			last_column = self.image_array[:,-1].reshape(-1,1)
			self.image_array = np.concatenate((self.image_array, last_column), axis=1)
		self.width = self.image_array.shape[1]

		if self.bidirectional_enabled:
			self.interleave_fw_bw()

	def interleave_fw_bw(self):
		# TODO: improve this code
		image_fw = self.image_array
		image_bw = np.fliplr(self.image_array)
		image_fw_bw = np.empty((self.image_array.shape[0], 2*self.image_array.shape[1]), dtype=self.image_array.dtype)
		for i in range(0,self.image_array.shape[1]):
			image_fw_bw[:,2*i] = image_fw[:,i]
			image_fw_bw[:,2*i+1] = image_bw[:,i]

		warnings.warn("Add check for the mult of 4....")

		self.image_array = image_fw_bw
	
class Alphabets(object):
	def __init__(self, alphabets_path, alphabets_size):
		self.alphabets_path = alphabets_path
		self.alphabets_size = alphabets_size
		self.alphabets = np.loadtxt(self.alphabets_path, dtype=np.str, delimiter='##')
		self.alphabets = np.append([""], self.alphabets)
		if len(self.alphabets) != self.alphabets_size:
			raise Exception("Wrong number of symbols in alphabet")

	def ReturnString(self, pred):
		length = pred[0]
		string = ""
		for i in range(1,int(length+1)):
			string += str(self.alphabets[pred[i]])
		return string
