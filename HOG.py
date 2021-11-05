import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import math
import os.path
class HOG:
	nbin = 9
	nr_b, nc_b = 2,2
	def __init__(self, nr_b = 2, nc_b = 2):
		self.nc_b = nc_b
		self.nr_b = nr_b
	def extractImg(self,class_no=1, image_no=1, type="training"):
		type = "training" if (type == "training") else "test"
		path = str(class_no) + "/" + str(class_no) + str(image_no) + '_'+type.capitalize()+".bmp"
		return plt.imread(path)
	def rgb2gray(self,rgb):
		if rgb.ndim != 3:
			return rgb
		return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])
	def GradientMandO(self,I):
		Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
		Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
		Ix = signal.correlate2d(I, Sx, mode='same')
		Iy = signal.correlate2d(I, Sy, mode='same')
		# I_mag: gradient magnitude
		I_mag=np.sqrt(Ix**2 + Iy**2) 
		# Gradient orientation
		nr, nc = I.shape
		Ipr = np.zeros(shape=(nr, nc))
		I_angle = np.zeros(shape=(nr, nc))
		for j in range(nr):
			for i in range(nc):
				if abs(Ix[j, i]) <= 0.0001 and abs(Iy[j, i]) <= 0.0001: # Both Ix and Iy areclose to zero
					I_angle[j, i] = 0.00
				else:
					Ipr[j, i] = math.atan(Iy[j,i]/(Ix[j,i]+np.finfo(float).eps)) #Compute the angle in radians
					I_angle[j, i] = Ipr[j, i]*180/math.pi # Compute the angle in degrees
					if Ix[j, i] < 0: # If Ix is negative, 180 degrees added
						I_angle[j, i] = 180+I_angle[j, i]
					if I_angle[j, i] < 0: # If the angle is negative,360 degrees added
						I_angle[j, i] = 360+I_angle[j, i]
		return (I_mag, I_angle)
	def HoG1(self,Im, Ip, nbin):
		ghist = np.zeros(shape=(1,nbin))
		[nr1, nc1] = Im.shape
		interval = np.round(180/nbin, 0)
		for i in range(nr1):
			for j in range(nc1):
				if Ip[i, j] > 180:
					Ip[i, j] = abs(Ip[i, j] - 360)
				index = int(np.int(Ip[i, j]/interval))
				if index >= nbin:
					index = index - 1
				ghist[0, index] += np.square(Im[i,j]) 
		return ghist
	def Histogram_Normalization(self,ihist):
		total_sum = np.sum(ihist)
		nhist = ihist / total_sum
		return nhist
	def getFeatureVec(self,I, I_mag, I_angle, nr_b, nc_b):
		nr, nc = I.shape
		nbin = 9
		nr_size = int(nr/nr_b)
		nc_size = int(nc/nc_b)
		Image_HoG = np.zeros(shape=(1, nbin*nr_b*nc_b))
		for i in range(nr_b):
			for j in range(nc_b):
				I_mag_block = I_mag[i*nr_size: (i+1)*nr_size, j*nc_size: (j+1)*nc_size]
				I_angle_block = I_angle[i*nr_size: (i+1)*nr_size, j*nc_size: (j+1)*nc_size]
				# HoG1 creates HoG histogram
				gh = self.HoG1(I_mag_block, I_angle_block, nbin)
				# Histogram_Normalization normalizes the input histogram gh
				ngh = self.Histogram_Normalization(gh)
				pos = j*nbin+i*nc_b*nbin
				Image_HoG[:, pos:pos+nbin] = ngh
		return Image_HoG
	def ClassifyTrainAndTestImages(self):
		if(os.path.exists("h1.npy") and os.path.exists("h2.npy")):
			h1 = np.load("h1.npy")
			h2 = np.load("h2.npy")
			print("h1 and h2 are retrieved")
		else:
			h1 = np.zeros(shape=(25, self.nbin*self.nr_b*self.nc_b)) # training data
			h2 = np.zeros(shape=(25, self.nbin*self.nr_b*self.nc_b)) # test data
			for i in range(1,6):
				for j in range(1,6):
					I = self.rgb2gray(self.extractImg(i,j,"training"))
					h1[(j-1)+(i-1)*5] = self.getFeatureVec(I,self.GradientMandO(I)[0],self.GradientMandO(I)[1],self.nr_b, self.nc_b)
					I = self.rgb2gray(self.extractImg(i,j,"test"))
					h2[(j-1)+(i-1)*5] = self.getFeatureVec(I,self.GradientMandO(I)[0],self.GradientMandO(I)[1],self.nr_b, self.nc_b)
			np.save("h1.npy",h1)
			np.save("h2.npy",h2)
			print("h1 and h2 are saved")
		if(os.path.exists("d1.npy") and os.path.exists("d2.npy") and os.path.exists("chi.npy")):
			d1 = np.load("d1.npy")
			d2 = np.load("d2.npy")
			chi = np.load("chi.npy")
			print("d1, d2 and chi are retrieved")
		else:
			d1 = np.zeros (shape=(25,25))
			d2 = d1.copy()
			chi = d2.copy()
			for i in range(25):
				for j in range(25):
					d1[i, j] = np.around(np.sum(np.abs(h2[i, :]-h1[j, :])),4)
					d2[i, j] = np.around(np.sum(np.square(np.abs(h2[i, :]-h1[j, :]))), 4)
					chi[i, j] = np.around(np.sum(np.square(np.abs(h2[i, :]-h1[j, :])) / (h2[i,:]+h1[j, :]+np.finfo(float).eps)), 4)
			np.save("d1.npy",d1)
			np.save("d2.npy",d2)
			np.save("chi.npy",chi)
		d1_min = np.argmin(d1,axis=1)
		d2_min = np.argmin(d2,axis=1)
		chi_min = np.argmin(chi,axis=1)
		acc = np.zeros(shape=(25))
		for i in range(5):
			for j in range(5):
				acc[j+5*i] = 5*i
		d1_res = d1_min - acc
		d2_res = d2_min - acc
		chi_res = chi_min - acc
		d1_acc=d2_acc=chi_acc = 0
		for i in d1_res:
			if(i>=0 and i<=4):
				d1_acc+=1
		d1_acc = d1_acc/d1_res.size
		for i in d2_res:
			if(i>=0 and i<=4):
				d2_acc+=1
		d2_acc = d2_acc/d2_res.size
		for i in chi_res:
			if(i>=0 and i<=4):
				chi_acc+=1
		chi_acc = chi_acc/chi_res.size
		print("L1-metric`s accuracy "+str(d1_acc)+"\n L2-metric`s accuracy "+str(d2_acc)+"\n Chi-square distance`s accuracy "+str(chi_acc))
		

obj = HOG(3,3)
obj.ClassifyTrainAndTestImages()

