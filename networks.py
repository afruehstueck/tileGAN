# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf
import sys
# NOTE: Do not import any application-specific modules here!
KERNEL_SIZE = 3

#----------------------------------------------------------------------------

def lerp(a, b, t): return a + (b - a) * t
def lerp_clip(a, b, t): return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
def cset(cur_lambda, new_cond, new_lambda): return lambda: tf.cond(new_cond, new_lambda, cur_lambda)

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
	if fan_in is None: fan_in = np.prod(shape[:-1])
	std = gain / np.sqrt(fan_in) # He init
	if use_wscale:
		wscale = tf.constant(np.float32(std), name='wscale')
		return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
	else:
		return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False):
	if len(x.shape) > 2:
		x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
	w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
	w = tf.cast(w, x.dtype)
	return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
	assert kernel >= 1 and kernel % 2 == 1
	w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
	w = tf.cast(w, x.dtype)
	#print('here is the issue')
	return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x):
	b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
	b = tf.cast(b, x.dtype)
	if len(x.shape) == 2:
		return x + b
	else:
		return x + tf.reshape(b, [1, -1, 1, 1])

#----------------------------------------------------------------------------
# Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.

def leaky_relu(x, alpha=0.2):
	with tf.name_scope('LeakyRelu'):
		alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
		return tf.maximum(x * alpha, x)

#----------------------------------------------------------------------------
# Nearest-neighbor upscaling layer.

def upscale2d(x, factor=2):
	assert isinstance(factor, int) and factor >= 1
	if factor == 1: return x
	with tf.variable_scope('Upscale2D'):
		s = x.shape
		x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
		x = tf.tile(x, [1, 1, 1, factor, 1, factor])
		x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
		return x

def conv2d_padded(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
	assert kernel >= 1 and kernel % 2 == 1
	w = get_weight([kernel, kernel, fmaps, x.shape[1].value], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
	#w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
	w = tf.cast(w, x.dtype)
	w = tf.transpose(w, perm=[0, 1, 3, 2])
	#print('here is the issue')
	return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID', data_format='NCHW')

#----------------------------------------------------------------------------
# Fused upscale2d + conv2d.
# Faster and uses less memory than performing the operations separately.

# def upscale2d_conv2d_padded(x, edge_1, edge_2, pad_w, axis, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
# 	assert kernel >= 1 and kernel % 2 == 1
# 	w = get_weight([kernel, kernel, fmaps, x.shape[1].value], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
# 	w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
# 	w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
# 	w = tf.cast(w, x.dtype)
#
# 	if axis == -1: #add edges to left and right, paddings to top and bottom
# 		paddings = tf.constant([[0, 0], [0, 0], [pad_w, pad_w], [0, 0]])
# 	else: #add edges to top and bottom, paddings to left and right
# 		paddings = tf.constant([[0, 0], [0, 0], [0, 0], [pad_w, pad_w]])
#
# 	padded_x = tf.concat([edge_1, x, edge_2], axis=axis) #add edges along correct dimension
# 	padded_x = tf.pad(padded_x, paddings, "CONSTANT") #add zero dimensions along the shape that doesn't have custom edges
#
# 	#print('padded input size: ', padded_x.shape)
# 	#print('weights shape:', w.shape)
# 	#print('targeted output shape: ', fmaps, padded_x.shape[2] * 2, padded_x.shape[3] * 2)
#
# 	output_shape = [tf.shape(x)[0], fmaps, padded_x.shape[2] * 2, padded_x.shape[3] * 2]
# 	result = tf.nn.conv2d_transpose(padded_x, w, output_shape, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
# 	#print('result shape before slice: ', result.shape)
# 	result = result[:, :, 2*pad_w:-2*pad_w, 2*pad_w:-2*pad_w]
# 	#print('result shape after slice: ', result.shape)
# 	return result
# 	#return tf.nn.conv2d_transpose(padded_x, w, output_shape, strides=[1, 1, 2, 2], padding='VALID', data_format='NCHW')
# 	#return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

def upscale2d_conv2d_padded2(x_padded, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
	assert kernel >= 1 and kernel % 2 == 1
	s = x_padded.shape
	w = get_weight([kernel, kernel, fmaps, s[1].value], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*s[1].value)
	w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
	w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
	w = tf.cast(w, x_padded.dtype)

	output_shape = [tf.shape(x_padded)[0], fmaps, s[2] * 2, s[3] * 2]
	#result = tf.nn.conv2d_transpose(padded_x, w, output_shape, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
	#print('result shape before slice: ', result.shape)
	#result = result[:, :, 2*pad_w:-2*pad_w, 2*pad_w:-2*pad_w]
	#print('result shape after slice: ', result.shape)
	return tf.nn.conv2d_transpose(x_padded, w, output_shape, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
	#return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Fused upscale2d + conv2d.
# Faster and uses less memory than performing the operations separately.

def upscale2d_conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
	assert kernel >= 1 and kernel % 2 == 1
	w = get_weight([kernel, kernel, fmaps, x.shape[1].value], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
	w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
	w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
	w = tf.cast(w, x.dtype)
	os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
	return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Box filter downscaling layer.

def downscale2d(x, factor=2):
	assert isinstance(factor, int) and factor >= 1
	if factor == 1: return x
	with tf.variable_scope('Downscale2D'):
		ksize = [1, 1, factor, factor]
		return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True

#----------------------------------------------------------------------------
# Fused conv2d + downscale2d.
# Faster and uses less memory than performing the operations separately.

def conv2d_downscale2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
	assert kernel >= 1 and kernel % 2 == 1
	w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
	w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
	w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
	w = tf.cast(w, x.dtype)
	return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.

def pixel_norm(x, epsilon=1e-8):
	with tf.variable_scope('PixelNorm'):
		return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

#----------------------------------------------------------------------------
# Minibatch standard deviation.

def minibatch_stddev_layer(x, group_size=4):
	with tf.variable_scope('MinibatchStddev'):
		group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
		s = x.shape                                             # [NCHW]  Input shape.
		y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
		y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
		y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
		y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
		y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
		y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
		y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
		y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [N1HW]  Replicate over group and pixels.
		return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.

#----------------------------------------------------------------------------
# Generator network used in the paper.

def G_paper(
	latents_in,                         # First input: Latent vectors [minibatch, latent_size].
	labels_in,                          # Second input: Labels [minibatch, label_size].
	num_channels        = 1,            # Number of output color channels. Overridden based on dataset.
	resolution          = 32,           # Output resolution. Overridden based on dataset.
	label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
	fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
	fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
	fmap_max            = 512,          # Maximum number of feature maps in any layer.
	latent_size         = None,         # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
	normalize_latents   = True,         # Normalize latent vectors before feeding them to the network?
	use_wscale          = True,         # Enable equalized learning rate?
	use_pixelnorm       = True,         # Enable pixelwise feature vector normalization?
	pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
	use_leakyrelu       = True,         # True = leaky ReLU, False = ReLU.
	dtype               = 'float32',    # Data type to use for activations and outputs.
	fused_scale         = True,         # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
	structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
	is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
	**kwargs):                          # Ignore unrecognized keyword args.

	#print('calling G_paper')

	resolution_log2 = int(np.log2(resolution))
	assert resolution == 2**resolution_log2 and resolution >= 4
	def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
	def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
	if latent_size is None: latent_size = nf(0)
	if structure is None: structure = 'linear' if is_template_graph else 'recursive'
	act = leaky_relu if use_leakyrelu else tf.nn.relu

	latents_in.set_shape([None, latent_size])
	labels_in.set_shape([None, label_size])
	combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)
	lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

	# Building blocks.
	def block(x, res): # res = 2..resolution_log2
		with tf.variable_scope('%dx%d' % (2**res, 2**res)):
			if res == 2: # 4x4
				if normalize_latents: x = pixel_norm(x, epsilon=pixelnorm_epsilon)
				with tf.variable_scope('Dense'):
					x = dense(x, fmaps=nf(res-1)*16, gain=np.sqrt(2)/4, use_wscale=use_wscale) # override gain to match the original Theano implementation
					x = tf.reshape(x, [-1, nf(res-1), 4, 4])
					x = PN(act(apply_bias(x)))
				with tf.variable_scope('Conv'):
					x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=KERNEL_SIZE, use_wscale=use_wscale))))
			else: # 8x8 and up
				if fused_scale:
					with tf.variable_scope('Conv0_up'):
						x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=KERNEL_SIZE, use_wscale=use_wscale))))
				else:
					x = upscale2d(x)
					with tf.variable_scope('Conv0'):
						x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=KERNEL_SIZE, use_wscale=use_wscale))))
				with tf.variable_scope('Conv1'):
					x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=KERNEL_SIZE, use_wscale=use_wscale))))
			return x
	def torgb(x, res): # res = 2..resolution_log2
		lod = resolution_log2 - res
		with tf.variable_scope('ToRGB_lod%d' % lod):
			return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

	####################### SIMPLIFIED GENERATOR - FOR TRAINED NETWORK #######################
	# x = combo_in
	# for res in range(2, resolution_log2 + 1):
	# 	x = block(x, res)
	#
	# images_out = torgb(x, res)
	##########################################################################################

	# Linear structure: simple but inefficient.
	if structure == 'linear':
		x = block(combo_in, 2)
		images_out = torgb(x, 2)
		for res in range(3, resolution_log2 + 1):
			lod = resolution_log2 - res
			x = block(x, res)
			img = torgb(x, res)
			images_out = upscale2d(images_out)
			with tf.variable_scope('Grow_lod%d' % lod):
				images_out = lerp_clip(img, images_out, lod_in - lod)

	# Recursive structure: complex but efficient.
	if structure == 'recursive':
		def grow(x, res, lod):
			y = block(x, res)
			img = lambda: upscale2d(torgb(y, res), 2**lod)
			if res > 2: img = cset(img, (lod_in > lod), lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1)), lod_in - lod), 2**lod))
			if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
			return img()
		images_out = grow(combo_in, 2, resolution_log2 - 2)

	assert images_out.dtype == tf.as_dtype(dtype)
	images_out = tf.identity(images_out, name='images_out')
	return images_out

#----------------------------------------------------------------------------
# Generator network not used in the paper.
def G_new(
	x_in,
	num_channels        = 1,            # Number of output color channels. Overridden based on dataset.
	resolution          = 32,           # Output resolution. Overridden based on dataset.
	in_res 				= 2,
	out_res 			= 9,
	latent_depth 		= 32,
	label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
	fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
	fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
	fmap_max            = 512,          # Maximum number of feature maps in any layer.
	latent_size         = None,         # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
	normalize_latents   = True,         # Normalize latent vectors before feeding them to the network?
	use_wscale          = True,         # Enable equalized learning rate?
	use_pixelnorm       = True,         # Enable pixelwise feature vector normalization?
	pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
	use_leakyrelu       = True,         # True = leaky ReLU, False = ReLU.
	dtype               = 'float32',    # Data type to use for activations and outputs.
	fused_scale         = True,         # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
	structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
	is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
	**kwargs):                          # Ignore unrecognized keyword args.

	#print('constrained combinator @target_res {}, latent_depth {}'.format(target_res, latent_depth))
	resolution_log2 = int(np.log2(resolution))

	assert in_res > 1 and out_res < 11
	assert resolution == 2**resolution_log2 and resolution >= 4
	def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
	def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
	#if latent_size is None: latent_size = nf(0)
	act = leaky_relu if use_leakyrelu else tf.nn.relu
	if in_res == 2:
		if latent_size is None: latent_size = [None, latent_depth]
	else:
		if latent_size is None: latent_size = [None, latent_depth, 2**(in_res - 1), 2**(in_res - 1)]

	x_in.set_shape(latent_size)

	# Building blocks.
	def block(x, res): # res = 2..resolution_log2
		with tf.variable_scope('%dx%d' % (2**res, 2**res)):
			if res == 2: # 4x4
				if normalize_latents: x = pixel_norm(x, epsilon=pixelnorm_epsilon)
				with tf.variable_scope('Dense'):
					x = dense(x, fmaps=nf(res-1)*16, gain=np.sqrt(2)/4, use_wscale=use_wscale) # override gain to match the original Theano implementation
					x = tf.reshape(x, [-1, nf(res-1), 4, 4])
					x = PN(act(apply_bias(x)))
				with tf.variable_scope('Conv'):
					x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=KERNEL_SIZE, use_wscale=use_wscale))))
			else: # 8x8 and up
				if fused_scale:
					with tf.variable_scope('Conv0_up'):
						#x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=KERNEL_SIZE, use_wscale=use_wscale))))
						x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=KERNEL_SIZE, use_wscale=use_wscale))))
				else:
					x = upscale2d(x)
					with tf.variable_scope('Conv0'):
						x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=KERNEL_SIZE, use_wscale=use_wscale))))
				with tf.variable_scope('Conv1'):
					x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=KERNEL_SIZE, use_wscale=use_wscale))))
			return x

	def torgb(x, res): # res = 2..resolution_log2
		lod = resolution_log2 - res
		with tf.variable_scope('ToRGB_lod%d' % lod):
			return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

	x = x_in

	for res in range(in_res, out_res + 1):
		x = block(x, res)

	images_out = torgb(x, res)

	return x, images_out

#----------------------------------------------------------------------------
# Generator network not used in the paper.
def G_stage_arbitrary_size(
	ABCD,
	num_channels        = 1,            # Number of output color channels. Overridden based on dataset.
	resolution          = 32,           # Output resolution. Overridden based on dataset.
	in_res 				= 2,
	out_res 			= 9,
	latent_depth 		= 32,
	label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
	fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
	fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
	fmap_max            = 512,          # Maximum number of feature maps in any layer.
	latent_size         = None,         # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
	normalize_latents   = True,         # Normalize latent vectors before feeding them to the network?
	use_wscale          = True,         # Enable equalized learning rate?
	use_pixelnorm       = True,         # Enable pixelwise feature vector normalization?
	pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
	use_leakyrelu       = True,         # True = leaky ReLU, False = ReLU.
	dtype               = 'float32',    # Data type to use for activations and outputs.
	fused_scale         = True,         # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
	structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
	is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
	**kwargs):                          # Ignore unrecognized keyword args.

	#print('constrained combinator @target_res {}, latent_depth {}'.format(target_res, latent_depth))
	resolution_log2 = int(np.log2(resolution))

	assert in_res > 1 and out_res < 10
	assert resolution == 2**resolution_log2 and resolution >= 4
	def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
	def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
	#if latent_size is None: latent_size = nf(0)
	act = leaky_relu if use_leakyrelu else tf.nn.relu
	if in_res == 2:
		if latent_size is None: latent_size = [None, 2*latent_depth]
	else:
		if latent_size is None: latent_size = [None, latent_depth, 2**(in_res - 1), 2**(in_res - 1)]

	print('incoming latent shape: ', latent_size)
	ABCD.set_shape(latent_size)

	# Building blocks.
	def block(x, res): # res = 2..resolution_log2
		with tf.variable_scope('%dx%d' % (2**res, 2**res)):
			if res == 2: # 4x4
				if normalize_latents: x = pixel_norm(x, epsilon=pixelnorm_epsilon)
				with tf.variable_scope('Dense'):
					x = dense(x, fmaps=nf(res-1)*16, gain=np.sqrt(2)/4, use_wscale=use_wscale) # override gain to match the original Theano implementation
					x = tf.reshape(x, [-1, nf(res-1), 4, 4])
					x = PN(act(apply_bias(x)))
				with tf.variable_scope('Conv'):
					x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=KERNEL_SIZE, use_wscale=use_wscale))))
			else: # 8x8 and up
				if fused_scale:
					with tf.variable_scope('Conv0_up'):
						x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=KERNEL_SIZE, use_wscale=use_wscale))))
				else:
					x = upscale2d(x)
					with tf.variable_scope('Conv0'):
						x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=KERNEL_SIZE, use_wscale=use_wscale))))
				with tf.variable_scope('Conv1'):
					x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=KERNEL_SIZE, use_wscale=use_wscale))))
			return x

	def block_padded(x, res): # res = 2..resolution_log2
		with tf.variable_scope('%dx%d' % (2**res, 2**res)):
			with tf.variable_scope('Conv0_up'):
				x = PN(act(apply_bias(upscale2d_conv2d_padded2(x, fmaps=nf(res-1), kernel=KERNEL_SIZE, use_wscale=use_wscale))))
			with tf.variable_scope('Conv1'):
				x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=KERNEL_SIZE, use_wscale=use_wscale))))
			return x

	def torgb(x, res): # res = 2..resolution_log2
		lod = resolution_log2 - res
		#print('lod of toRGB', lod)
		with tf.variable_scope('ToRGB_lod%d' % lod):
			return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

	x = ABCD

	for res in range(in_res, out_res + 1):
		x = block_padded(x, res)

	#print('shape after final block: ', x.shape)
	images_out = torgb(x, res)
	return x, images_out

#----------------------------------------------------------------------------
# Discriminator network used in the paper.
def D_paper(
	images_in,                          # Input: Images [minibatch, channel, height, width].
	num_channels        = 1,            # Number of input color channels. Overridden based on dataset.
	resolution          = 32,           # Input resolution. Overridden based on dataset.
	label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
	fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
	fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
	fmap_max            = 512,          # Maximum number of feature maps in any layer.
	use_wscale          = True,         # Enable equalized learning rate?
	mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
	dtype               = 'float32',    # Data type to use for activations and outputs.
	fused_scale         = True,         # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
	structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
	is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
	**kwargs):                          # Ignore unrecognized keyword args.

	resolution_log2 = int(np.log2(resolution))
	assert resolution == 2**resolution_log2 and resolution >= 4
	def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
	if structure is None: structure = 'linear' if is_template_graph else 'recursive'
	act = leaky_relu

	images_in.set_shape([None, num_channels, resolution, resolution])
	images_in = tf.cast(images_in, dtype)
	lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

	# Building blocks.
	def fromrgb(x, res): # res = 2..resolution_log2
		with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
			return act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, use_wscale=use_wscale)))
	def block(x, res): # res = 2..resolution_log2
		with tf.variable_scope('%dx%d' % (2**res, 2**res)):
			if res >= 3: # 8x8 and up
				with tf.variable_scope('Conv0'):
					x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=KERNEL_SIZE, use_wscale=use_wscale)))
				if fused_scale:
					with tf.variable_scope('Conv1_down'):
						x = act(apply_bias(conv2d_downscale2d(x, fmaps=nf(res-2), kernel=KERNEL_SIZE, use_wscale=use_wscale)))
				else:
					with tf.variable_scope('Conv1'):
						x = act(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=KERNEL_SIZE, use_wscale=use_wscale)))
					x = downscale2d(x)
			else: # 4x4
				if mbstd_group_size > 1:
					x = minibatch_stddev_layer(x, mbstd_group_size)
				with tf.variable_scope('Conv'):
					x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=KERNEL_SIZE, use_wscale=use_wscale)))
				with tf.variable_scope('Dense0'):
					x = act(apply_bias(dense(x, fmaps=nf(res-2), use_wscale=use_wscale)))
				with tf.variable_scope('Dense1'):
					x = apply_bias(dense(x, fmaps=1+label_size, gain=1, use_wscale=use_wscale))
			return x

	# Linear structure: simple but inefficient.
	if structure == 'linear':
		img = images_in
		x = fromrgb(img, resolution_log2)
		for res in range(resolution_log2, 2, -1):
			lod = resolution_log2 - res
			x = block(x, res)
			img = downscale2d(img)
			y = fromrgb(img, res - 1)
			with tf.variable_scope('Grow_lod%d' % lod):
				x = lerp_clip(x, y, lod_in - lod)
		combo_out = block(x, 2)

	# Recursive structure: complex but efficient.
	if structure == 'recursive':
		def grow(res, lod):
			x = lambda: fromrgb(downscale2d(images_in, 2**lod), res)
			if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
			x = block(x(), res); y = lambda: x
			if res > 2: y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
			return y()
		combo_out = grow(2, resolution_log2 - 2)

	assert combo_out.dtype == tf.as_dtype(dtype)
	scores_out = tf.identity(combo_out[:, :1], name='scores_out')
	labels_out = tf.identity(combo_out[:, 1:], name='labels_out')
	return scores_out, labels_out

#----------------------------------------------------------------------------
