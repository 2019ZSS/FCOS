# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @description: code origin https://github.com/stevewongv/DSC-PyTorch/edit/master/irnn.py
'''

import os 
import torch
import math
import cupy
from os import pipe
from torch import nn
import torch.nn.functional as F


dirname, filename = os.path.split(os.path.abspath(__file__))

IRNNForward = open(os.path.join(dirname, 'IRNN_Forward_cuda.cu'), 'r').read()

IRNNBackward = open(os.path.join(dirname, 'IRNN_Backward_cuda.cu'), 'r').read()


class Stream:
	ptr = torch.cuda.current_stream().cuda_stream

# 报错，改用cupy.memoize
# @cupy.util.memoize(for_each_device=True)
@cupy.memoize(for_each_device=True)
def cunnex(strFunction):
	return cupy.cuda.compile_with_cache(globals()[strFunction]).get_function(strFunction)


class IRNNFunction(torch.autograd.Function):

	def __init__(self):
		super(IRNNFunction, self).__init__()
		
	@staticmethod
	def forward(self, input_feature, weight_up, weight_right, weight_down, weight_left, bias_up, bias_right, bias_down, bias_left):
		assert torch.cuda.is_available()
		input_feature = input_feature.cuda()
		assert(input_feature.is_contiguous() == True)
		assert(weight_left.is_contiguous() == True)
		assert(weight_right.is_contiguous() == True)
		assert(weight_down.is_contiguous() == True)

		assert(weight_up.is_contiguous() == True)
		assert(bias_left.is_contiguous() ==True)
		assert(bias_right.is_contiguous() == True)
		assert(bias_up.is_contiguous() == True)
		assert(bias_down.is_contiguous() == True)

		output_left = input_feature.clone()
		output_right = input_feature.clone()
		output_up = input_feature.clone()
		output_down = input_feature.clone()

		if input_feature.is_cuda == True:
			n = input_feature.nelement()
			cuda_num_threads = 1024
			cunnex('IRNNForward')(
				grid=tuple([ int((n +  cuda_num_threads - 1) / cuda_num_threads ), 1, 1 ]),
				block=tuple([ cuda_num_threads , 1, 1 ]),
				args=[
					input_feature.data_ptr(),
					
					weight_up.data_ptr(), 
					weight_right.data_ptr(),
					weight_down.data_ptr(),
					weight_left.data_ptr(),

					bias_up.data_ptr(),
					bias_right.data_ptr(),
					bias_down.data_ptr(),
					bias_left.data_ptr(),
					
					output_up.data_ptr(), 
					output_right.data_ptr(),
					output_down.data_ptr(), 
					output_left.data_ptr(),
					
					input_feature.size(1),
					input_feature.size(2),
					input_feature.size(3),
					n],
				stream=Stream
			)
		elif input_feature.is_cuda == False:
			raise NotImplementedError()
		
		
		self.save_for_backward(input_feature,weight_up,weight_right,weight_down,weight_left,output_up,output_right,output_down,output_left)

		return output_up,output_right,output_down,output_left
	
	@staticmethod
	def backward(self, grad_output_up,grad_output_right,grad_output_down,grad_output_left):

		input_feature,weight_up,weight_right,weight_down,weight_left,output_up,output_right,output_down,output_left = self.saved_tensors
		# print(weight_left)
		if grad_output_up.is_contiguous() != True:
			grad_output_up = grad_output_up.contiguous()
		if grad_output_right.is_contiguous() != True:
			grad_output_right = grad_output_right.contiguous()
		if grad_output_down.is_contiguous() != True:
			grad_output_down = grad_output_down.contiguous()
		if grad_output_left.is_contiguous() != True:
			grad_output_left = grad_output_left.contiguous()

		# init gradient of input_feature
		grad_input = torch.zeros_like(input_feature)
		# init gradient map of weights
		grad_weight_up_map = torch.zeros_like(input_feature)
		grad_weight_right_map = torch.zeros_like(input_feature)
		grad_weight_down_map = torch.zeros_like(input_feature)
		grad_weight_left_map = torch.zeros_like(input_feature)
		# init gradient of weights
		grad_weight_left = torch.zeros_like(weight_left)
		grad_weight_right = torch.zeros_like(weight_left)
		grad_weight_up = torch.zeros_like(weight_left)
		grad_weight_down = torch.zeros_like(weight_left)

		grad_bias_up_map = torch.zeros_like(input_feature)
		grad_bias_right_map = torch.zeros_like(input_feature)
		grad_bias_down_map = torch.zeros_like(input_feature)
		grad_bias_left_map = torch.zeros_like(input_feature)

		if input_feature.is_cuda == True:
			
			n = grad_input.nelement()

			cuda_num_threads = 1024		
			cunnex('IRNNBackward')(
				grid=tuple([ int((n +  cuda_num_threads - 1) / cuda_num_threads), 1, 1 ]),
				block=tuple([ cuda_num_threads , 1, 1 ]),
				args=[ 
					grad_input.data_ptr(),

					grad_weight_up_map.data_ptr(),
					grad_weight_right_map.data_ptr(),
					grad_weight_down_map.data_ptr(),
					grad_weight_left_map.data_ptr(),

					grad_bias_up_map.data_ptr(),  
					grad_bias_right_map.data_ptr(),
					grad_bias_down_map.data_ptr(),
					grad_bias_left_map.data_ptr(),

					weight_up.data_ptr(),
					weight_right.data_ptr(),
					weight_down.data_ptr(),
					weight_left.data_ptr(),
					
					grad_output_up.data_ptr(),
					grad_output_right.data_ptr(),
					grad_output_down.data_ptr(),
					grad_output_left.data_ptr(),
					
					output_up.data_ptr(),					
					output_right.data_ptr(),					
					output_down.data_ptr(),
					output_left.data_ptr(),

					input_feature.size(1),
					input_feature.size(2),
					input_feature.size(3),
					n],
				stream=Stream
			)
			# print(grad_weight_left_map,"<-- grad weight map")

			grad_bias_up = torch.zeros_like(weight_left).reshape(weight_left.size(0))
			grad_bias_right = torch.zeros_like(weight_left).reshape(weight_left.size(0))
			grad_bias_down = torch.zeros_like(weight_left).reshape(weight_left.size(0))
			grad_bias_left = torch.zeros_like(weight_left).reshape(weight_left.size(0))

			grad_weight_left  = grad_weight_left_map.sum(2).sum(2).sum(0).resize_as_(grad_weight_left)
			grad_weight_right = grad_weight_right_map.sum(2).sum(2).sum(0).resize_as_(grad_weight_left)
			grad_weight_up    = grad_weight_up_map.sum(2).sum(2).sum(0).resize_as_(grad_weight_left)
			grad_weight_down  = grad_weight_down_map.sum(2).sum(2).sum(0).resize_as_(grad_weight_left)

			grad_bias_up    = grad_bias_up_map.sum(2).sum(2).sum(0).resize_as_(grad_bias_up)
			grad_bias_right = grad_bias_right_map.sum(2).sum(2).sum(0).resize_as_(grad_bias_up)
			grad_bias_down  = grad_bias_down_map.sum(2).sum(2).sum(0).resize_as_(grad_bias_up)
			grad_bias_left  = grad_bias_left_map.sum(2).sum(2).sum(0).resize_as_(grad_bias_up)
			
		elif input_feature.is_cuda == False:
			raise NotImplementedError()

		
		return grad_input, grad_weight_up,grad_weight_right,grad_weight_down,grad_weight_left,grad_bias_up, grad_bias_right, grad_bias_down, grad_bias_left


class Spacial_IRNN(nn.Module):
	'''
	如果将循环权重矩阵初始化为单位矩阵，这些网络将很容易训练，并且擅长建模长期依赖关系。
	这意味着在初始化时，梯度会以全强度向后传播，因此将以这种方式初始化的ReLU RNN称为IRNN，
	对于现实世界的语言建模任务，它的表现几乎和LSTM一样好，对于内存问题比LSTM更好。
	'''
	def __init__(self, in_channels, alpha=1.0):
		super(Spacial_IRNN, self).__init__()

		self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
		self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
		self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
		self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
		self.left_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
		self.right_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
		self.up_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
		self.down_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
	
	def forward(self, input):
		'''
		Args:
		input: tensor [batch_size, C, H, W]
		return: tuple[tensor[B, C, H, W]]
		output_up,output_right,output_down, output_left
		'''
		return IRNNFunction.apply(input, self.up_weight.weight, self.right_weight.weight, self.down_weight.weight, self.left_weight.weight, self.up_weight.bias, self.right_weight.bias, self.down_weight.bias, self.left_weight.bias)


class IRCNN(nn.Module):
	
	def __init__(self, in_channels, out_channels, alpha=1.0):
		super(IRCNN, self).__init__()
		# reduce channels
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
		self.sirnn1 = Spacial_IRNN(out_channels, alpha)
		self.conv2 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
		self.srinn2 = Spacial_IRNN(out_channels, alpha)
		self.conv3 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
		self.relu = nn.ReLU(True)
	
	def forward(self, input):
		assert torch.cuda.is_available()
		out = torch.cat(self.sirnn1(self.conv1(input.cuda())), dim=1)
		out = torch.cat(self.srinn2(self.conv2(out)), dim=1)
		return self.relu(self.conv3(out))


if __name__ == '__main__':
	# x = torch.rand((4, 256, 64, 64))
	# s_irnn = Spacial_IRNN(256)
	# y = s_irnn(x.cuda())
	# for v in y:
	# 	print(v.shape)
	# print(x.device)
	assert torch.cuda.is_available()
	torch.manual_seed(0)
	torch.cuda.manual_seed(0)
	torch.cuda.manual_seed_all(0)
	input = torch.rand((4, 2048, 32, 32))
	print(input.device)
	print(input.shape)
	ircnn = IRCNN(2048, 256).cuda()
	ircnn = torch.torch.nn.DataParallel(ircnn)
	print(ircnn)
	out = ircnn(input)
	print(out.shape)
	print(input.device)