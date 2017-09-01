import numpy as np
from PIL import Image
import sys
from os import path
import math
import cv2

caffe_root = r'/home/yangkunlin/storage/PSPNet/PSPNet-master/'
sys.path.insert(0, path.join(caffe_root, 'python'))
import caffe

def test():
	mean_r = 123.68
	mean_g = 116.779
	mean_b = 103.939
	data_list ="/home/yangkunlin/storage/PSPNet/PSPNet-master/evaluation/list/003wang_simple"
	file_object = open(data_list)
	model_deploy = 'prototxt/pspnet101_cityscapes_713.prototxt' #
	model_weights = 'model/pspnet101_cityscapes.caffemodel'		#
	save_root = 'mc_result/003wang'
	net = caffe.Net(model_deploy, model_weights, caffe.TEST)
	for line in file_object:
		line = line.replace("\n","")
		print (line)
		img = cv2.imread('/home/yangkunlin/storage/PSPNet/PSPNet-master/evaluation/list/'+line)  #
		sp = img.shape
		# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		# print ("Image shape: ")
		# sp = img.shape
		# print sp
		im_mean = np.zeros([713,713,3],float)
		im_mean[:,:,0] = mean_b
		im_mean[:,:,1] = mean_g
		im_mean[:,:,2] = mean_r
		img = cv2.resize(img, (713, 713))
		img = img.astype(np.float)
		img = img - im_mean
		img = img.transpose(2, 0, 1)
		img = np.array([img])	
		net.blobs['data'].data[...].reshape(1,3,713,713)
		net.blobs['data'].data[...] = img
		net.forward()
		predict =  net.blobs['conv6_interp'].data.copy()
		print (predict.shape)
		
		b=np.zeros((713,713, 3))
		for j in range(0,713):
			for k in range(0,713):
				x = -1
				label = 0
				for i in range(0,19):
					if predict[0][i][j][k]>x:
						x=predict[0][i][j][k]
						label=i
				if label == 0:
					b[j][k][0]=b[j][k][1]=b[j][k][2]=0
				if label == 1:
					b[j][k][0]=b[j][k][1]=b[j][k][2]=255
				if label == 2:
					b[j][k][0]=255
					b[j][k][1]=b[j][k][2]=0
				if label == 3:
					b[j][k][0]=156
					b[j][k][1]=102
					b[j][k][2]=31
				if label == 4:
					b[j][k][0]=b[j][k][1]=255
					b[j][k][2]=0
				if label == 5:
					b[j][k][0]=255
					b[j][k][1]=153
					b[j][k][2]=18
				if label == 6:
					b[j][k][0]=255
					b[j][k][1]=127
					b[j][k][2]=80
				if label == 7:
					b[j][k][0]=255
					b[j][k][1]=0
					b[j][k][2]=255
				if label == 8:
					b[j][k][0]=0
					b[j][k][1]=255
					b[j][k][2]=0
				if label == 9:
					b[j][k][0]=0
					b[j][k][1]=255
					b[j][k][2]=255
				if label == 10:
					b[j][k][0]=8
					b[j][k][1]=46
					b[j][k][2]=84
				if label == 11:
					b[j][k][0]=107
					b[j][k][1]=142
					b[j][k][2]=35
				if label == 12:
					b[j][k][0]=255
					b[j][k][1]=215
					b[j][k][2]=0
				if label == 13:
					b[j][k][0]=255
					b[j][k][1]=125
					b[j][k][2]=64
				if label == 14:
					b[j][k][0]=255
					b[j][k][1]=227
					b[j][k][2]=132
				if label == 15:
					b[j][k][0]=85
					b[j][k][1]=102
					b[j][k][2]=0
				if label == 16:
					b[j][k][0]=188
					b[j][k][1]=143
					b[j][k][2]=143
				if label == 17:
					b[j][k][0]=160
					b[j][k][1]=82
					b[j][k][2]=45
				if label == 18:
					b[j][k][0]=218
					b[j][k][1]=112
					b[j][k][2]=214
			# b[j][k][0]=b[j][k][1]=b[j][k][2]=label*10
		print ("path: ")
		pic_path = path.join(save_root,line)
		print (pic_path)   #
		cv2.imwrite(pic_path,b)
		img2 = cv2.imread (pic_path)
		img2 = cv2.resize(img2, (sp[1], sp[0]))
		cv2.imwrite(pic_path,img2)


if __name__ == '__main__':
    test()