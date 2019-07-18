import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pickle
import h5py
import os
from joblib import dump, load

default_batch_size = 1000 #number of consecutively processed latents
target_directory = 'data'

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'create descriptors and cluster tool')
	parser.add_argument('path', metavar='F', type=str, help='file path to network pickle')
	parser.add_argument('num_latents', metavar='N', type=int, help='number of latents in database', default=200000)
	parser.add_argument('t_size', metavar='T', type=int, help='size of padded output descriptors', default=20)
	parser.add_argument('num_clusters', metavar='C', type=int, help='number of clusters', default=12)
	parser.add_argument('network_name', metavar='NN', type=str, help='network name')

	args = parser.parse_args()

	########################################################################################################################
	#PARSE ARGUMENTS
	########################################################################################################################

	filename = args.path
	num_latents = args.num_latents
	t_size = args.t_size
	num_clusters = args.num_clusters
	network_name = args.network_name

	directory = os.path.join(target_directory, network_name)
	########################################################################################################################
	#CREATE DIRECTORY
	########################################################################################################################
	#create directory, if it doesn't exist
	if not os.path.exists(directory):
		os.makedirs(directory)

	# LAYER      2    3    4    5    6    7    8   9
	depths = [512, 512, 512, 512, 512, 256, 128, 64, 32]  # depth of intermediate network output
	minibatch_size = 8

	channels = 3

	print('Initializing TensorFlow...')
	tf.InteractiveSession()

	random_state = np.random.RandomState()

	########################################################################################################################
	# LOAD NETWORK
	########################################################################################################################
	try:
		with open(filename, 'rb') as file:
			G, D, Gs = pickle.load(file)
	except TypeError:
		print('You must specify the path to a valid trained ProGAN network.')

	#saving only generator network for later use
	with open(os.path.join(directory, '{}_network.pkl'.format(network_name)), 'wb') as file:
		pickle.dump(Gs, file, protocol=pickle.HIGHEST_PROTOCOL)

	########################################################################################################################
	# WRITE INFO FILE
	########################################################################################################################
	f = open(os.path.join(directory, 'data.txt'), "w+")
	f.write("num_latents=%d\r\n" % num_latents)
	f.write("descriptor size=%d\r\n" % t_size)
	f.write("num clusters=%d\r\n" % num_clusters)
	f.close()

	########################################################################################################################
	# GENERATE LATENTS AND REPRESENTATIONS
	########################################################################################################################
	strdesc = ''
	strdesc += str(t_size)
	hdf5_path = os.path.join(directory, '{}_descriptors.hdf5'.format(network_name))
	create_descriptors = not os.path.exists(hdf5_path)

	hdf5_file = h5py.File(hdf5_path)

	total_desc_size = t_size * t_size * channels
	print('total descriptor length: ', total_desc_size)
	if create_descriptors:
		descriptors = np.zeros((default_batch_size, total_desc_size), dtype=np.uint8)

		chunk_descriptor_shape = (default_batch_size, total_desc_size)
		chunk_latents_shape = (default_batch_size, 512)

		chunk_descriptors = np.empty(chunk_descriptor_shape)
		chunk_latents = np.empty(chunk_latents_shape)

		descriptors_dataset = hdf5_file.create_dataset("descriptors", (0, total_desc_size), maxshape=(None, total_desc_size), dtype='uint8', chunks=chunk_descriptor_shape)
		latents_dataset = hdf5_file.create_dataset("latents", (0, 512), maxshape=(None, 512), dtype='float32', chunks=chunk_latents_shape)
		chunk_count = 0

		for run in tqdm(np.arange(num_latents // default_batch_size)):
			latents = np.random.randn(default_batch_size, *Gs.input_shape[1:]).astype(np.float32)
			labels = np.zeros([latents.shape[0], 0], np.float32)

			padded_images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=1, out_mul=127.5, out_add=127.5, out_dtype=np.uint8)

			batch_index = 0

			# generate descriptors
			for image in padded_images:
				image = np.squeeze(image)
				if image.shape[0] == 3:
					image = np.rollaxis(image, 0, 3)
				im = Image.fromarray(image)

				im.thumbnail((t_size, t_size))
				descriptors[batch_index, :] = np.ravel(im)
				batch_index += 1

			descriptors_dataset.resize((chunk_count + default_batch_size, total_desc_size))
			descriptors_dataset[chunk_count:] = descriptors
			latents_dataset.resize((chunk_count + default_batch_size, 512))
			latents_dataset[chunk_count:] = latents
			chunk_count += default_batch_size

	########################################################################################################################
	# GENERATE CLUSTERS
	########################################################################################################################
	hdf5_file = h5py.File(hdf5_path)
	latents = hdf5_file["latents"].value
	padded_descriptors = hdf5_file["descriptors"].value
	print('descriptors shape: {}'.format(padded_descriptors.shape))

	kmeans_path = os.path.join(directory, '{}_kmeans.joblib'.format(network_name))

	if os.path.isfile(kmeans_path):
		kmeans = load(kmeans_path)
		print('loaded KMeans!')
	else:
		print('calculating {}-means for {} descriptors'.format(num_clusters, len(padded_descriptors)))
		kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(padded_descriptors)
		dump(kmeans, kmeans_path)
		if create_descriptors:
			hdf5_file.create_dataset("clusters", data=kmeans.labels_)
		print('dumped KMeans!')

	hdf5_file.close()

	def get_averaged_cluster_centers(clustering, num_clusters):
		"""Generates plot to view the cluster centroids."""
		average_images = np.zeros((num_clusters, t_size, t_size, 3), dtype=np.uint8)
		for c in range(num_clusters):
			pixData = clustering.cluster_centers_[c, :]
			pixImg = np.reshape(pixData, (t_size, t_size, 3))
			average_images[c] = pixImg

		return average_images

	def get_upsampled_cluster_centers(clustering, num_clusters, latents, descriptors):
		"""Generates plot to view the cluster centroids."""
		m = 1
		cluster_images = np.zeros((num_clusters, Gs.output_shape[3], Gs.output_shape[3], 3), dtype=np.uint8)
		for c in range(num_clusters):
			distances = clustering.transform(descriptors)[:, c]
			ind = np.argsort(distances)[::][:m]  # n closest latents
			closest_latents = latents[ind]
			labels = np.zeros([closest_latents.shape[0], 0], np.float32)

			outputs = Gs.run(closest_latents, labels, minibatch_size=minibatch_size, num_gpus=1, out_mul=127.5, out_add=127.5, out_dtype=np.uint8)
			cluster_images[c] = np.rollaxis(outputs[0], 0, 3)
		return cluster_images

	average_images = get_averaged_cluster_centers(kmeans, num_clusters)
	cluster_images = get_upsampled_cluster_centers(kmeans, num_clusters, latents, padded_descriptors)

	clusters = kmeans.labels_
	count = np.bincount(clusters)

	hdf5_cluster_path = os.path.join(directory, '{}_{}_clusters.hdf5'.format(network_name, num_clusters))
	hdf5_file_clusters = h5py.File(hdf5_cluster_path, "w")
	for cluster in range(num_clusters):
		this_cluster = np.argwhere(clusters == cluster)
		this_descriptors = padded_descriptors[clusters == cluster]
		print('cluster {}'.format(cluster))

		assert (len(this_descriptors) == count[cluster])
		print(this_descriptors.shape)
		hdf5_file_clusters.create_dataset("{}".format(cluster), data=this_cluster)
		distances = kmeans.transform(this_descriptors)[:, cluster]
		indices = np.argsort(distances)[::][:]
		print('max index=', np.argmax(indices))
		max_dist = np.amax(distances)
		print(max_dist)
		probabilities = 1 - np.clip(distances / 3000, 0, 1)
		probabilities /= probabilities.sum()
		print('probabilities between {} and {}'.format(np.amin(probabilities), np.amax(probabilities)))
		cdf = [probabilities[0]]
		for i in np.arange(1, len(probabilities)):
			cdf.append(cdf[-1] + probabilities[i])
		assert (len(cdf) == count[cluster])
		hdf5_file_clusters.create_dataset("{}_cdf".format(cluster), data=cdf)

	hdf5_file_clusters.create_dataset("images", data=cluster_images)
	hdf5_file_clusters.create_dataset("averages", data=average_images)
	hdf5_file_clusters.close()