from multiprocessing.managers import BaseManager
from sklearn.cluster import KMeans
import os
import glob
import tensorflow as tf
import numpy as np
import time
import h5py
from bisect import bisect
from spinner import Spinner
from joblib import load
import pickle
from PIL import Image
import hnswlib
import signal
import socket

## LOOKUP TABLE FOR NETWORK DEPTH
#LEVEL = [  2,   3,   4,   5,   6,   7,   8,   9,   10]
#SIZES = [  4,   8,  16,  32,  64, 128, 256, 512, 1024]
DEPTHS = [512, 512, 512, 512, 512, 256, 128,  64,   32]


class TFManager:
	"""
		TFManager handles the server-side TensorFlow processing of TileGAN
	"""
	def __init__(self):
		self.mergeLevel = 2
		self.latentSize = 2
		self.tSize = 0
		self.numClusters = 0
		self.chunkShape = 24  # has to be multiple of 4!
		self.useANN = True

		self.undoDepth = 5
		self.undoCount = 0

		self.height = 0
		self.width = 0
		self.latentDepth = 0

		self.clusterSamples = None

		self.latentClusters = []
		self.latentCDFs = []
		self.latentImages = []
		self.averageImages = []
		self.dominantClusterColors = []

		self.output = None

		self.networkPath = None
		self.networkA = None
		self.sessionA = None
		self.GsA = None
		self.graphA = None
		self.networkB = None
		self.sessionB = None
		self.GsB = None
		self.graphB = None
		self.networkC = None
		self.sessionC = None
		self.GsC = None
		self.graphC = None

		self.latentLookup = None
		self.descriptorLookup = None
		self.clusterLookup = None
		self.annNbrs = None
		self.kmeans = None
		self.intermLatents = None
		self.intermLatentGrid = None
		self.latentList = None

		self.clusterGrid = None
		self.descriptorGrid = None
		self.undoStack = None

		self.dataset = None
		self.guidanceImage = None

		availDatasets, _ = self.findDatasets()
		print('datasets found in data directory: ', availDatasets)
		if len(availDatasets) > 0:
			self.dataset = availDatasets[0]
			self.initDataset(availDatasets[0])


	def findDatasets(self, searchDir=None):
		"""
		find all available directories in search space
		"""
		if searchDir is None:
			searchDir = 'data'
		datasetFolders = [f.name for f in os.scandir(searchDir) if f.is_dir()]
		return datasetFolders, self.dataset

	# ---------------------------------------------------------------------------------------------------------------------------------------------------------
	def initDataset(self, datasetName):
		"""
		initialize dataset with name dataPath
		"""
		dataPaths = self.parseDataset(datasetName)
		if dataPaths is None:
			return

		networkPath, descriptorsPath, clustersPath, kmeansPath, annPath, metadata = dataPaths
		self.dataset = datasetName
		self.networkPath = networkPath

		self.initNetworks()
		self.initLatentClusters(clustersPath)
		self.loadLatents(descriptorsPath)
		if self.useANN:
			self.initANN(annPath)
		else:
			pass
			# self.initNNRecords()
		self.initKMeans(kmeansPath)
		self.dominantClusterColors = []


	def parseDataset(self, datasetFolder):
		"""
		extract required files from dataset directory and load them to memory
		"""
		searchdirs = []
		searchdirs += ['data']

		targetDir = None
		for searchdir in searchdirs:
			dir = os.getcwd() if searchdir == '' else os.path.join(os.getcwd(), searchdir)
			dir = os.path.join(dir, str(datasetFolder))
			if os.path.isdir(dir):
				targetDir = dir
				break

		requiredFiles = True
		fnames = glob.glob(os.path.join(targetDir, '*descriptors.hdf5'))
		if len(fnames) > 0:
			descriptors = fnames[0]
		else:
			requiredFiles = False
			print('no descriptor file found in {}'.format(targetDir))

		fnames = glob.glob(os.path.join(targetDir, '*network.pkl'))
		if len(fnames) > 0:
			network = fnames[0]
		else:
			requiredFiles = False
			print('no network file found in {}'.format(targetDir))

		if not requiredFiles:
			print('didn\'t find required files in directory: {}'.format(targetDir))
			raise ValueError('Could not find valid dataset.')
			return None

		fnames = glob.glob(os.path.join(targetDir, '*clusters.hdf5'))
		if len(fnames) > 0:
			clusters = fnames[0]
		else:
			clusters = os.path.join(targetDir, '{}_clusters.hdf5'.format(datasetFolder))

		fnames = glob.glob(os.path.join(targetDir, '*kmeans.joblib'))
		if len(fnames) > 0:
			kmeans = fnames[0]
		else:
			kmeans = os.path.join(targetDir, '{}_kmeans.joblib'.format(datasetFolder))

		fnames = glob.glob(os.path.join(targetDir, '*ann.bin'))
		if len(fnames) > 0:
			ann = fnames[0]
		else:
			ann = os.path.join(targetDir, '{}_ann.bin'.format(datasetFolder))

		dataPath = os.path.join(targetDir, 'data.txt')

		metadata = {}

		if os.path.isfile(dataPath):
			with open(dataPath) as f:
				for line in f:
					c = line.rstrip('\n').split('=')
					metadata[c[0]] = int(c[1])
		return network, descriptors, clusters, kmeans, ann, metadata

	def setMergeLevel(self, level, latentSize=-1):
		"""
		update merge level and latent size of network
		"""
		self.mergeLevel = level
		if latentSize < 0:
			self.latentSize = 2 ** level
		else:
			self.latentSize = latentSize

		print('new level: {} | decrusted width: {}'.format(level, self.latentSize))
		self.initNetworks()

	def closeSessions(self):
		self.sessionA.close()
		self.sessionB.close()
		self.sessionC.close()

	def initNetworks(self):
		"""
		initialize the networks from the specified network paths.
		"""
		t1 = time.time()

		inResA = 2
		outResA = self.mergeLevel
		depthA = int(DEPTHS[inResA - 2])

		inResB = self.mergeLevel + 1
		depthB = int(DEPTHS[inResB - 2])
		inSizeB = 2 ** (inResB - 1)

		self.latentDepth = depthB
		print('Creating session and loading graph for stage A...')
		self.sessionA = tf.Session()

		#! loading Gs 2x/3x is not ideal
		with self.sessionA.as_default():
			with open(self.networkPath, 'rb') as file:
				Gs = pickle.load(file)

				size = Gs.output_shape[-1]
				channels = Gs.output_shape[1]

				self.outRes = int(np.log2(Gs.output_shape[-1]))
			kwargs = {'in_res': inResA, 'out_res': outResA, 'latent_depth': depthA, 'label_size': 0, 'num_channels': channels, 'resolution': size}
			self.GsA = Gs.clone_and_update("GsA", kwargs=kwargs, func='networks.G_new')
			self.graphA = tf.get_default_graph()

		print('Creating session and loading graph for stage B...')
		self.sessionB = tf.Session()
		with self.sessionB.as_default():
			with open(self.networkPath, 'rb') as file:
				Gs = pickle.load(file)
			outRes = int(np.log2(Gs.output_shape[-1]))
			kwargs = {'in_res': inResB, 'out_res': outRes, 'latent_depth': depthB, 'latentSize': [None, depthB, inSizeB, inSizeB], 'label_size': 0, 'num_channels': channels, 'resolution': size}
			self.GsB = Gs.clone_and_update("GsB", kwargs=kwargs, func='networks.G_new')
			self.graphB = tf.get_default_graph()

		print('Creating session and loading graph for stage C...') #USING A SEPARATE STAGE C TO GENERATE SINGLE LATENT OUTPUTS
		self.sessionC = tf.Session()
		with self.sessionC.as_default():
			with open(self.networkPath, 'rb') as file:
				Gs = pickle.load(file)
			outRes = int(np.log2(Gs.output_shape[-1]))
			kwargs = {'in_res': inResB, 'out_res': outRes, 'latent_depth': depthB, 'latentSize': [None, depthB, inSizeB, inSizeB], 'label_size': 0, 'num_channels': channels, 'resolution': size}
			self.GsC = Gs.clone_and_update("GsC", kwargs=kwargs, func='networks.G_new')
			self.graphC = tf.get_default_graph()

		t2 = time.time()
		print('initializing networks {:10.2f}s '.format(t2 - t1))

	def initLatentClusters(self, clustersPath):
		"""
		load the clustering from the specified h5py file.
		"""
		t1 = time.time()
		self.latentClusters = []
		self.latentCDFs = []
		hdf5_file = h5py.File(clustersPath, "r")
		self.latentImages = hdf5_file["images"].value
		self.averageImages = hdf5_file["averages"].value
		self.numClusters = len(self.latentImages)
		print('found {} clusters'.format(self.numClusters))
		for i in range(self.numClusters):
			self.latentClusters.append(hdf5_file["{}".format(i)])
			self.latentCDFs.append(hdf5_file["{}_cdf".format(i)])
		t2 = time.time()
		print('initializing latentClusters {:10.2f}s '.format(t2 - t1))

	def getDominantClusterColors(self):
		"""
		returns the dominant image color for each of the cluster images
		"""
		if not self.dominantClusterColors:
			for image in self.averageImages:
				pixels = np.float32(image.reshape(-1, 3))

				n_colors = 5
				clustering = KMeans(n_clusters=n_colors).fit(pixels)
				count = np.bincount(clustering.labels_)
				sorted_indices = np.argsort(count)[::-1]
				sorted_colors = clustering.cluster_centers_[sorted_indices, :]
				self.dominantClusterColors.append(sorted_colors[0].astype(np.uint8))

		return self.dominantClusterColors

	def getLatentImages(self):
		return self.latentImages

	def getLatentAverages(self):
		return self.averageImages

	def loadLatents(self, descriptorsPath):
		"""
		load the latent bank from the specified h5py
		"""
		hdf5_file = h5py.File(descriptorsPath, "r")
		self.descriptorLookup = hdf5_file["descriptors"].value
		self.latentLookup = hdf5_file["latents"].value
		self.clusterLookup = hdf5_file["clusters"].value

		descriptor = self.descriptorLookup[0]

		self.tSize = int(np.sqrt(len(descriptor)/3))
		print('tSize is {}'.format(self.tSize))
		print('loaded {} latents'.format(len(self.latentLookup)))

	def initANN(self, annPath, force=False):
		"""
		load or initialize approximate nearest neighbor finding in latent database. ANN searching is performed using HNSW
		"""
		t1 = time.time()

		dims = len(self.descriptorLookup[0])

		self.annNbrs = hnswlib.Index(space='l2', dim=dims)

		if os.path.isfile(annPath) and not force:
			print('found existing ANN index, loading...')
			self.annNbrs.load_index(annPath)
			return

		spinner = Spinner()
		spinner.start()
		print('... creating new ANN index for {} descriptors of dimension {}'.format(len(self.descriptorLookup), dims))

		# HNSW Parameter settings
		ef_construction = 2000  # reasonable range: 100-2000
		ef_search = 2000  # reasonable range: 100-2000 #if higher, better recall but longer retrieval time
		M = 100  # reasonable range: 5-100 (higher = more accuracy, longer retrieval time)

		self.annNbrs.init_index(max_elements=len(self.descriptorLookup), ef_construction=ef_construction, M=M)
		self.annNbrs.add_items(self.descriptorLookup, np.arange(len(self.descriptorLookup)))
		self.annNbrs.set_ef(ef_search)  # higher ef leads to better accuracy, but slower search
		self.annNbrs.save_index(annPath)

		spinner.stop()
		t2 = time.time()
		print('initializing init_ann_records {:10.2f}s '.format(t2 - t1))

	def initKMeans(self, kmeansPath):
		"""
		load or create KMeans from latent database
		"""
		#global kmeans, descriptorLookup
		t1 = time.time()
		print('clustering latents...')
		spinner = Spinner()
		spinner.start()

		if os.path.isfile(kmeansPath):
			self.kmeans = load(kmeansPath)
			print('loaded KMeans!')
		else:
			print('no saved KMeans found, creating from scratch...')
			print('calculating {}-means for {} descriptors'.format(self.numClusters, len(self.descriptorLookup)))
			self.kmeans = KMeans(n_clusters=self.numClusters, random_state=0).fit(self.descriptorLookup)

		spinner.stop()
		#debug does loading work?
		t2 = time.time()
		print('clustering latents {:10.2f}s '.format(t2 - t1))

	def getLeftPadding(self):
		return (2 ** self.mergeLevel - self.latentSize) // 2

	def getAdjacentClusters(self, y, x):
		"""
		find clusters of neighboring tiles
		"""
		tiles_y = int(self.clusterGrid.shape[0] / self.latentSize)
		tiles_x = int(self.clusterGrid.shape[1] / self.latentSize)
		adjacent_clusters = []
		if x > 0:
			adjacent_clusters.append(self.clusterGrid[y, x - 1])
		if x < tiles_x - 1:
			adjacent_clusters.append(self.clusterGrid[y, x + 1])
		if y > 0:
			adjacent_clusters.append(self.clusterGrid[y - 1, x])
		if y < tiles_y - 1:
			adjacent_clusters.append(self.clusterGrid[y + 1, x])
		return adjacent_clusters

	def MRFLatents(self, threshold = 0.15, maxIter = 25, lambda_v = 1, lambda_l = 0.5, lambda_c = 0.1):
		"""
		run MRF optimization on latent field
		"""
		t1 = time.time()

		outputH = self.output.shape[1]
		outputW = self.output.shape[2]
		print('output dimensions are: {}x{}'.format(outputH, outputW))

		tSize = self.tSize
		w = self.latentSize
		pad = self.getLeftPadding()

		gridY = int(self.clusterGrid.shape[0])
		gridX = int(self.clusterGrid.shape[1])
		tilesY = int(gridY / self.latentSize)
		tilesX = int(gridX / self.latentSize)

		outSize = 2 ** (self.outRes - self.mergeLevel)
		imgPad = pad * (tSize // (w + 2 * pad))
		gSize = tSize - 2 * imgPad

		N = 7  # pick a reasonable number of candidates
		M = 3  # pick a reasonable number of top-picks

		E_m = 0.0 #euclidean distance between output and guidance image
		E_n = 1.0 #sum of dissimilarity terms
		outputComparisonImage = None
		guidanceComparisonImage = None

		if self.guidanceImage is not None: #E_m is only relevant if guidance image exists
			guidanceComparisonImage = self.guidanceImage
			guidancePIL = Image.fromarray(self.guidanceImage)
			guidancePIL = guidancePIL.resize((outputW, outputH))
			outputComparisonImage = np.array(guidancePIL)
			outputComparisonImage = np.rollaxis(outputComparisonImage, 2, 0) #adjust axes to fit output
			E_m = np.linalg.norm(self.output - outputComparisonImage)
		else:
			outArr = np.rollaxis(self.output, 0, 3)
			outPIL = Image.fromarray(outArr)
			outPIL = outPIL.resize((gSize * tilesX, gSize * tilesY))
			guidanceComparisonImage = np.array(outPIL)
		totalEnergy = E_m + E_n

		i = 0
		# until energy threshold is reached, update field and calculate MRF
		while totalEnergy > threshold and i < maxIter:
			i = i + 1
			#randomly sample a latent tile
			x = np.random.randint(1, gridX - 2*w) #currently not handling border cases
			y = np.random.randint(1, gridY - 2*w)
			print('randomly sampling at ({}, {})'.format(x, y))

			gX = int(x * guidanceComparisonImage.shape[1]/gridX)
			gY = int(y * guidanceComparisonImage.shape[0]/gridY)
			tile = guidanceComparisonImage[gY:gY + tSize, gX:gX + tSize, :]
			print(tile.shape)
			descriptor = np.ravel(tile)  # find descriptor region in guidance image

			if(len(descriptor) < tSize * tSize * tile.shape[2]): #verify descriptor size
				continue

			indices, distances = self.annNbrs.knn_query(descriptor, N)

			candidates = np.squeeze(self.latentLookup[indices]) #get N best matches for descriptor region
			candidateClusters = np.squeeze(self.clusterLookup[indices])
			intermediateCandidates = self.calculateIntermediateLatents(candidates)

			#put candidate tiles into latent field and pass through second generator / check if possibly only descriptor region should be replaced
			candidateOutputs = []
			candidateDistances = []
			candidateGrid = np.copy(self.intermLatentGrid)

			print('calculating candidate outputs...')
			maxDiff = outputW * outputH * 3 * 64
			for n in range(N):
				intermLatent = intermediateCandidates[n]
				candidateGrid[:, :, y:y + w, x:x + w] = intermLatent[:, pad:pad + w, pad:pad + w]

				candidateOutput = self.calculateOutputImage(candidateGrid, start=(0, 0), end=(0, 0), updateAll=True)
				candidateOutputs.append(candidateOutput)

			if self.guidanceImage is not None: #E_m calculation only for guidance images
				distance = np.linalg.norm(candidateOutput - outputComparisonImage) / maxDiff
				candidateDistances.append(distance)

				print('calculated candidate distances: ', candidateDistances)
				# find N top candidates using E_m
				sortedByDistance = np.argsort(candidateDistances)
				bestCandidateIndices = sortedByDistance[:M]
				print('best candidate indices: ', bestCandidateIndices)
				bestCandidates = candidates[bestCandidateIndices]
				E_ms = np.array(candidateDistances)[bestCandidateIndices]
			else:
				bestCandidates = candidates[:M]
				E_ms = np.zeros(M)

			E_ns = []

			#calculate dissimilarity terms for each candidate
			for c, candidate in enumerate(bestCandidates):
				distances_v = 0
				distances_l = 0
				distance_c = 0

				#### D_v ####
				if lambda_v > 0:
					#get neighbors in 4-neighborhood, calculating dissimilarity of edge region
					neighborDescriptors = []

					neighborDescriptors.append(guidanceComparisonImage[gY - gSize:gY - gSize + tSize, gX:gX + tSize, :])  # TOP
					neighborDescriptors.append(guidanceComparisonImage[gY + gSize:gY + gSize + tSize, gX:gX + tSize, :])  # BOTTOM
					neighborDescriptors.append(guidanceComparisonImage[gY:gY + tSize, gX - gSize:gX - gSize + tSize, :])  # LEFT
					neighborDescriptors.append(guidanceComparisonImage[gY:gY + tSize, gX - gSize:gX + gSize + tSize, :])  # RIGHT
					candidateOutput = candidateOutputs[c]
					#resize output to guidance image size
					candidatePIL = Image.fromarray(np.rollaxis(candidateOutput, 0, 3))
					candidatePIL = candidatePIL.resize((guidanceComparisonImage.shape[1], guidanceComparisonImage.shape[0]))
					candidateComparisonImage = np.array(candidatePIL)

					candidateDescriptors = []
					candidateDescriptors.append(candidateComparisonImage[gY - gSize:gY - gSize + tSize, gX:gX + tSize, :])  # TOP
					candidateDescriptors.append(candidateComparisonImage[gY + gSize:gY + gSize + tSize, gX:gX + tSize, :])  # BOTTOM
					candidateDescriptors.append(candidateComparisonImage[gY:gY + tSize, gX - gSize:gX - gSize + tSize, :])  # LEFT
					candidateDescriptors.append(candidateComparisonImage[gY:gY + tSize, gX - gSize:gX + gSize + tSize, :])  # RIGHT

					maxDiff = tSize * tilesY * tSize * tilesX * 3 * 64 #fix range to normalize to and clamp to range
					#calculate distances for all 4 neighbors
					distances_v = [ np.linalg.norm(neighborDescriptors[i] - candidateDescriptors[i]) for i in range(4) ]
					distances_v = np.array(distances_v) / maxDiff

					print('distances_v: ', distances_v)
				#### D_l ####
				if lambda_l > 0:
					latentEdges = []
					latentEdges.append(self.intermLatentGrid[:, :, y - w, x:x + w].flatten()) # TOP
					latentEdges.append(self.intermLatentGrid[:, :, y + w, x:x + w].flatten()) # BOTTOM
					latentEdges.append(self.intermLatentGrid[:, :, y:y + w, x - w].flatten()) # LEFT
					latentEdges.append(self.intermLatentGrid[:, :, y:y + w, x + w].flatten()) # RIGHT

					intermLatent = intermediateCandidates[c]
					candidateEdges = []
					candidateEdges.append(intermLatent[:, pad - 1, pad:pad + w].flatten()) # TOP
					candidateEdges.append(intermLatent[:, pad + w, pad:pad + w].flatten()) # BOTTOM
					candidateEdges.append(intermLatent[:, pad:pad + w, pad - 1].flatten()) # LEFT
					candidateEdges.append(intermLatent[:, pad:pad + w, pad + w].flatten()) # RIGHT

					maxDiff = len(latentEdges[0]) * 0.5 #fix range to normalize to and clamp to range
					# calculate distances for all 4 neighbors
					distances_l = [np.linalg.norm(latentEdges[i] - candidateEdges[i]) for i in range(4)]
					distances_l = np.array(distances_l) / maxDiff
					print('distances_l: ', distances_l)

				#### D_c ####
				if lambda_c > 0:
					adjacentClusters = self.getAdjacentClusters(y, x)
					candidateCluster = candidateClusters[c]
					numSame = np.count_nonzero(adjacentClusters == candidateCluster)
					distance_c = 1.0 - float(numSame / len(adjacentClusters))
					print('distance_c: ', distance_c)

				D_v = lambda_v * sum(distances_v) #visual dissimilarity
				D_l = lambda_l * sum(distances_l) #latent dissimilarity
				D_c = lambda_c * distance_c #cluster membership
				E_n = D_v + D_l + D_c
				E_ns.append(E_n)

			print('E_ms: ', E_ms)
			print('E_ns: ', E_ns)
			totalEnergies = E_ms + E_ns
			#pick candidate with minimal energy
			sortedByEnergy = np.argsort(totalEnergies)
			selectedIdx = sortedByEnergy[0]
			if totalEnergies[selectedIdx] < totalEnergy: #only pick tile if energy goes down

				totalEnergy = totalEnergies[selectedIdx]
				print('totalEnergy: ', totalEnergy)
				intermLatent = intermediateCandidates[selectedIdx]
				candidateGrid[:, :, y : y + w, x : x + w] = intermLatent[:, pad:pad + w, pad:pad + w]

				self.intermLatentGrid = candidateGrid
				self.output = candidateOutputs[selectedIdx]

		t2 = time.time()
		print('improving latents {:10.2f}s '.format(t2 - t1))
		return self.output


	def getUpsampled(self, image):
		"""
		get a tile-per-tile arrangement of latent based on similarity to a guidance image
		"""
		t1 = time.time()

		tSize = self.tSize
		latentSize = self.latentSize

		#if necessary, remove alpha channel
		if image.shape[2] > 3:
			image = image[:, :, :3]

		imgH = image.shape[0]
		imgW = image.shape[1]
		print('input image shape: {}'.format(image.shape))

		channels = 3 if len(image.shape) > 2 else 1

		descSize = tSize * tSize * channels

		totalPad = (2 ** self.mergeLevel) - latentSize
		totalImgPad = totalPad * (tSize // (2 ** self.mergeLevel))
		gSize = tSize - totalImgPad

		#extract width and height of tiling
		self.height = (imgH - totalImgPad) // gSize
		self.width = (imgW - totalImgPad) // gSize
		print('number of tiles: ({}x{})'.format(self.height, self.width))

		self.guidanceImage = image[:self.height*gSize, :self.width*gSize, :]

		tileDescriptors = np.zeros((self.height * self.width, descSize))
		self.clusterGrid = np.zeros((self.height * self.latentSize, self.width * self.latentSize), dtype=np.uint8)

		for y in range(self.height):
			for x in range(self.width):
				tile = image[y * gSize:y * gSize + tSize, x * gSize:x * gSize + tSize, :]
				tileDescriptors[y * self.width + x, :] = np.ravel(tile)

		spinner = Spinner()
		spinner.start()
		num_options = 3
		print('finding nearest neighbors for {} descriptors of length {}'.format(tileDescriptors.shape[0], tileDescriptors.shape[1]))
		# extract indices of nearest neighbor
		if self.useANN:  # use ANN
			all_indices, all_distances = self.annNbrs.knn_query(tileDescriptors, num_options)
		else:  # use KNN
			#all_distances, all_indices = nnNbrs.kneighbors(tileDescriptors)
			pass

		print('predicting clusters')
		spinner.stop()

		print('generating latent list')
		latentList = []
		for i in range(self.height * self.width): #! currently picking random latent
			indices = all_indices[i, :]
			randomIdx = np.random.randint(len(indices))
			index = int(indices[randomIdx])
			latentList.append(self.latentLookup[index])
			y = i // self.width
			x = i % self.width
			self.clusterGrid[y*self.latentSize:(y+1)*self.latentSize, x*self.latentSize:(x+1)*self.latentSize] = int(self.clusterLookup[index])

		self.latentList = np.asarray(latentList)

		t2 = time.time()
		outputImage = self.getOutputFromLatents(self.latentList)

		gridH = self.height * latentSize
		gridW = self.width * latentSize

		t3 = time.time()
		print('processing nearest neighbors {:10.2f}s | processing grid {:10.4f}s'.format(t2 - t1, t3 - t2))

		saveImageOnServer=False
		if saveImageOnServer:
			import pyvips
			saveArray = np.copy(outputImage)

			if saveArray.shape[0] == 3:
				saveArray = np.rollaxis(saveArray, 0, 3)
			height, width, bands = saveArray.shape
			linear = saveArray.reshape(width * height * bands)
			vi = pyvips.Image.new_from_memory(linear, width, height, bands, 'uchar')
			vi.write_to_file('output.jpg')

		return outputImage, None, (gridH, gridW, latentSize, self.mergeLevel), self.undoCount

	def pasteLatents(self, sampleLatent, targetX, targetY, targetW, targetH, sourceX, sourceY, mode='identical'):
		"""
		cloning sample latent to larger region in the texture
		"""
		latentSize = self.latentSize
		pad = self.getLeftPadding()
		t1 = time.time()
		gridH = self.intermLatentGrid.shape[2]
		gridW = self.intermLatentGrid.shape[3]

		centerOffset = latentSize // 2
		print('target at [{}, {}], region size: ({}x{})'.format(targetX, targetY, targetW, targetH))

		#default: use source latent from latent grid - override this if mode is not 'identical'
		sourceX = min(max(centerOffset, sourceX), gridW - (latentSize - centerOffset))  # constrain sourceX to grid dims
		sourceY = min(max(centerOffset, sourceY), gridH - (latentSize - centerOffset))  # constrain sourceY to grid dims
		sourceLatent = self.intermLatentGrid[:, :, sourceY - centerOffset:sourceY - centerOffset + latentSize, sourceX - centerOffset:sourceX - centerOffset + latentSize]

		numSimilar = 5

		#override which latent is used
		if mode == 'similar':
			pilImg = Image.fromarray(sampleLatent)
			pilImg.thumbnail((self.tSize, self.tSize))
			sourceDescriptor = np.asarray(pilImg)

			if self.useANN:  # use ANN
				indices, distances = self.annNbrs.knn_query(np.ravel(sourceDescriptor).reshape(1, -1), numSimilar)
			else:  # use KNN
				distances, indices = self.nnNbrs.kneighbors(np.ravel(sourceDescriptor).reshape(1, -1))

			nearestLatents = self.latentLookup[indices]
			similarLatents = self.calculateIntermediateLatents(np.squeeze(nearestLatents))
			print('nearestLatents size: {}'.format(similarLatents.shape))
		elif mode == 'cluster':
			clusterIndex = self.clusterGrid[sourceY, sourceX]
			cluster = self.latentClusters[clusterIndex]
			cdf = self.latentCDFs[clusterIndex]
			print('cluster mode, using cluster {}'.format(clusterIndex))

		for xPos in range(0, targetW, latentSize):
			for yPos in range(0, targetH, latentSize):
				x_start = targetX + xPos
				y_start = targetY + yPos
				x_end = min(x_start + latentSize, targetX+targetW)
				y_end = min(y_start + latentSize, targetY+targetH)

				if mode == 'similar':
					# pick random latent from similar and decrust
					randomIdx = np.random.randint(numSimilar)
					sourceLatent = similarLatents[randomIdx:randomIdx+1, :, pad:pad+latentSize, pad:pad+latentSize]
				elif mode =='cluster':
					randomIdx = bisect(cdf, np.random.random())
					# pick random latent from similar and decrust
					sameClusterLatent = self.latentLookup[cluster[randomIdx]]
					intermLatent = self.calculateIntermediateLatents(sameClusterLatent)
					sourceLatent = intermLatent[:, :, pad:pad + latentSize, pad:pad + latentSize]
					self.clusterGrid[y_start:y_end, x_start:x_end] = clusterIndex

				# define whether latent is at canvas edge and needs to be cropped on right or bottom
				r = max((targetX+targetW), x_start + latentSize) - (targetX+targetW) #! simplify
				b = max((targetY+targetH), y_start + latentSize) - (targetY+targetH)
				print('y: [{}, {}], x: [{}, {}], cutRB: [{}, {}]'.format(y_start, y_end, x_start, x_end, r, b))
				self.intermLatentGrid[:, :, y_start:y_end, x_start:x_end] = sourceLatent[:, :, :latentSize - b, :latentSize - r]

		roi = 2 * latentSize
		outputImage = self.calculateOutputImage(self.intermLatentGrid, start=(max(targetY - roi, 0), max(targetX - roi, 0)), end=(min(targetY + targetH + 2 * roi, gridH), min(targetX + targetW + 2 * roi, gridW)), updateAll=False)
		t2 = time.time()
		print('pasting latent {:10.2f}s'.format(t2 - t1))
		return outputImage, self.undoCount

	def saveLatents(self):
		"""
		saving latents to file - ! load latents from file
		"""
		from pathlib import Path
		string = 'savelatents'
		gridH = self.intermLatentGrid.shape[2]
		gridW = self.intermLatentGrid.shape[3]
		timestr = time.strftime("%m_%d_%H%M")
		np.save(str(Path.home()) +'\Desktop\{}_{}x{}latents_{}.npy'.format(string, gridH, gridW, timestr), self.intermLatentGrid)


	def initLatentList(self, h, w, repeat=False):
		"""
		initialize the latent grid with a grid of random latent vectors
		"""
		self.height = h
		self.width = w

		self.clusterGrid = np.zeros((self.height*self.latentSize, self.width*self.latentSize), dtype=np.uint8)

		if repeat:
			randomIdx = np.random.randint(len(self.latentLookup))
			latent = self.latentLookup[randomIdx]
			cluster = self.clusterLookup[randomIdx]
			latentList = np.tile(latent, (self.height * self.width))
			self.clusterGrid = np.tile(cluster, (self.height*self.latentSize, self.width*self.latentSize))
		else:
			randomIdxs = np.random.randint(len(self.latentLookup), size=self.height * self.width)
			latentList = np.asarray([ self.latentLookup[randomIdx] for randomIdx in randomIdxs ])
			clusters = np.asarray([ self.clusterLookup[randomIdx] for randomIdx in randomIdxs ])

			for x in np.arange(self.width):
				for y in np.arange(self.height):
					self.clusterGrid[y*self.latentSize:(y+1)*self.latentSize, x*self.latentSize:(x+1)*self.latentSize] = clusters[y*w+x]

		print('initialized latent list of size: ', latentList.shape)
		return latentList

	def calculateIntermediateLatents(self, latents):
		"""
		process input latent vectors in {latents} using networkA and return intermediate latent vectors
		"""
		if len(latents.shape) == 1:
			latents = np.expand_dims(latents, axis=0)
		if len(latents.shape) > 2:
			latents = np.squeeze(latents)

		with self.graphA.as_default():
			with self.sessionA.as_default():
				intermediateLatents, _ = self.GsA.run(latents, in_res=2, out_res=self.mergeLevel, latent_depth=DEPTHS[0], minibatch_size=32, num_gpus=1, out_dtype=np.float32)

		return intermediateLatents

	def getOutputFromLatents(self, latentList, updateUndoStack=True):
		"""
		return the merged output based on the specified latent list
		"""
		if self.undoStack is None or self.undoStack.shape[1] != len(latentList):
			print('initialize undo stack...')
			self.initUndoStack()

		if updateUndoStack:
			self.putOnUndoStack(self.latentList)

		intermLatents = self.calculateIntermediateLatents(latentList)

		#! make decrusting (more) flexible
		latentSize = self.latentSize
		pad = self.getLeftPadding()
		self.intermLatentGrid = np.zeros((1, self.latentDepth, self.height * latentSize, self.width * latentSize))
		print('intermLatentGrid has shape', self.intermLatentGrid.shape)

		for y in range(self.height):
			for x in range(self.width):
				self.intermLatentGrid[:, :, y * latentSize:(y + 1) * latentSize, x * latentSize:(x + 1) * latentSize] = intermLatents[y * self.width + x, :, pad:pad+latentSize, pad:pad+latentSize]

		return self.calculateOutputImage(self.intermLatentGrid, updateAll=True)

	def initUndoStack(self):
		self.undoStack = np.zeros((self.undoDepth, self.height * self.width, 512))
		self.undoCount = 0

	def perturbLatent(self, posX, posY, sourceX, sourceY, alpha, randomLatent=False, fromSamples=False, useCDF=True):
		"""
		add latent in small quantities to current position in order to morph from one latent to another
		"""
		t1 = time.time()

		latentSize = self.latentSize
		pad = self.getLeftPadding()
		gridH = self.intermLatentGrid.shape[2]
		gridW = self.intermLatentGrid.shape[3]

		centerOffset = latentSize // 2
	
		sourceX = min(max(centerOffset, sourceX), gridW - (latentSize - centerOffset))  # constrain sourceX to grid dims
		sourceY = min(max(centerOffset, sourceY), gridH - (latentSize - centerOffset))  # constrain sourceY to grid dims
		intermLatent = self.intermLatentGrid[:, :, sourceY - centerOffset:sourceY - centerOffset + latentSize, sourceX - centerOffset:sourceX - centerOffset + latentSize]

		if randomLatent:
			clusterIndex = 0 #! random cluster no longer available
			if fromSamples and self.clusterSamples is not None:
				latent = self.clusterSamples[clusterIndex]
			else:
				cluster = self.latentClusters[clusterIndex]
				if useCDF:
					cdf = self.latentCDFs[clusterIndex]
					randomIdx = bisect(cdf, np.random.random())
				else:
					randomIdx = np.random.randint(len(cluster))
	
				randLatentIdx = cluster[randomIdx]
				latent = self.latentLookup[randLatentIdx]

				t2 = time.time()
				randIntermLatent = self.calculateIntermediateLatents(latent)

				# decrust random latent
				intermLatent = randIntermLatent[:, :, pad:pad + latentSize, pad:pad + latentSize]

		# get top left and bottom right from posX/Y (at center of click)
		x_start = max(posX - centerOffset, 0)
		y_start = max(posY - centerOffset, 0)
		x_end = min(posX - centerOffset + latentSize, gridW)  # weird workaround for odd latent sizes
		y_end = min(posY - centerOffset + latentSize, gridH)

		# define whether latent is at canvas edge and needs to be cropped on left, top, right, or bottom
		l = abs(min(0, posX - centerOffset))
		t = abs(min(0, posY - centerOffset))
		r = max(gridW, posX - centerOffset + latentSize) - gridW  # weird workaround for odd latent sizes
		b = max(gridH, posY - centerOffset + latentSize) - gridH

		assert not r < 0 and not b < 0

		self.intermLatentGrid[:, :, y_start:y_end, x_start:x_end] = (1-alpha) * self.intermLatentGrid[:, :, y_start:y_end, x_start:x_end] + alpha*intermLatent[:, :, t:latentSize - b, l:latentSize - r]

		roi = 2 * latentSize
		returnImage = self.calculateOutputImage(self.intermLatentGrid, start=(max(y_start - roi, 0), max(x_start - roi, 0)), end=(min(y_end + 2 * roi, gridH), min(x_end + 2 * roi, gridW)), updateAll=False)
		t3 = time.time()
		return returnImage, self.undoCount

	def putLatent(self, posX, posY, clusterIndex, fromSamples=False, useCDF=True, updateUndoStack=True):
		"""
		put a new latent of class {latent_class} at position {x_in}, {y_in} in the intermLatentGrid
		"""
		t1 = time.time()

		latentSize = self.latentSize
		pad = self.getLeftPadding()
		gridH = self.intermLatentGrid.shape[2]
		gridW = self.intermLatentGrid.shape[3]

		centerOffset = latentSize // 2

		# get top left and bottom right from posX/Y (at center of click)
		x_start = max(posX - centerOffset, 0)
		y_start = max(posY - centerOffset, 0)
		x_end = min(posX - centerOffset + latentSize, gridW)  # weird workaround for odd latent sizes
		y_end = min(posY - centerOffset + latentSize, gridH)

		# define whether latent is at canvas edge and needs to be cropped on left, top, right, or bottom
		l = abs(min(0, posX - centerOffset))
		t = abs(min(0, posY - centerOffset))
		r = max(gridW, posX - centerOffset + latentSize) - gridW  # weird workaround for odd latent sizes
		b = max(gridH, posY - centerOffset + latentSize) - gridH

		assert not r < 0 and not b < 0

		if fromSamples and self.clusterSamples is not None:
			randLatent = self.clusterSamples[clusterIndex]
			self.clusterGrid[y_start:y_end, x_start:x_end] = clusterIndex
		else:
			cluster = self.latentClusters[clusterIndex]
			if useCDF:
				cdf = self.latentCDFs[clusterIndex]
				randIdx = bisect(cdf, np.random.random())
			else:
				randIdx = np.random.randint(len(cluster))

			randLatentIdx = cluster[randIdx]
			randLatent = self.latentLookup[randLatentIdx]
			self.clusterGrid[y_start:y_end, x_start:x_end] = self.clusterLookup[randLatentIdx]

		t2 = time.time()
		randIntermLatent = self.calculateIntermediateLatents(randLatent)
		#decrust random latent
		randIntermLatent = randIntermLatent[:, :, pad:pad+latentSize, pad:pad+latentSize]

		if self.latentList is not None:
			self.latentList[(y_start//latentSize)*self.width+(x_start//latentSize)] = randLatent
			if updateUndoStack:
				self.putOnUndoStack(self.latentList)

		self.intermLatentGrid[:, :, y_start:y_end, x_start:x_end] = randIntermLatent[:, :, t:latentSize-b, l:latentSize-r]

		roi = 2 * latentSize
		returnImage = self.calculateOutputImage(self.intermLatentGrid, start=(max(y_start - roi, 0), max(x_start - roi, 0)), end=(min(y_end + 2 * roi, gridH), min(x_end + 2 * roi, gridW)), updateAll=False)
		t3 = time.time()
		return returnImage, self.undoCount

	def undo(self):
		"""
		undo the latest 'put latent' operation by pulling the previous status of intermLatentGrid from undoStack. This is currently a bit unsatisfactory as we cannot undo all kinds of latent operations.
		"""
		print('undoing last change...')
		self.latentList = self.undoStack[-2]
		self.undoStack = np.concatenate((self.undoStack[0:1], self.undoStack[:-1]), axis=0)

		undoCount = max(0, self.undoCount - 1)
		return self.getOutputFromLatents(self.latentList, updateUndoStack=False), undoCount

	def putOnUndoStack(self, latentList):
		self.undoStack = np.concatenate((self.undoStack[1:], np.expand_dims(latentList, axis=0)), axis=0)
		self.undoCount = min(self.undoCount + 1, self.undoDepth)

	def calculateUnmergedOutputImage(self, intermLatentGrid, mergeSize=0):
		"""
		calculate the output image from the grid of latents in {intermLatentGrid}
		"""

		if mergeSize == 0:
			mergeSize = self.latentSize

		gridH = (self.height * self.latentSize) // mergeSize
		gridW = (self.width * self.latentSize) // mergeSize

		#compile intermediate latents as list instead of grid
		intermLatentsList = np.zeros((gridH*gridW, self.latentDepth, mergeSize, mergeSize)) #it's necessary to recalculate latentsList here because user may have dropped latent in overlapping region
		s = list(intermLatentsList.shape)
		for y in np.arange(gridH):
			for x in np.arange(gridW):
				intermLatentsList[y*gridW+x, :, :, :] = intermLatentGrid[:, :, y*mergeSize:(y+1)*mergeSize, x*mergeSize:(x+1)*mergeSize]

		#feed list of intermediate latents through second network
		with self.graphC.as_default():
			with self.sessionC.as_default():
				if [s[1], mergeSize, mergeSize] != self.GsC.input_shape[1:]:  # only update if input shape changed
					self.GsC.update_latent_size(mergeSize, mergeSize)
				_, outputs = self.GsC.run(intermLatentsList, in_res=self.mergeLevel + 1, out_res=self.outRes, latent_size=[None, s[1], mergeSize, mergeSize], latent_depth=s[1], minibatch_size=32, num_gpus=1, out_mul=127.5, out_add=127.5, out_dtype=np.uint8)

		outSize = outputs.shape[2]

		output = np.zeros((3, outSize*gridH, outSize*gridW), np.uint8)

		#merge outputs into grid shape
		for y in range(gridH):
			for x in range(gridW):
				output[:, y * outSize:(y + 1) * outSize, x * outSize:(x + 1) * outSize] = outputs[y * gridW + x]

		return np.squeeze(output)

	def calculateOutputImage(self, intermLatentGrid, start=(0, 0), end=(0, 0), updateAll=False):
		"""
		calculate the output image from the grid of latents in {intermLatentGrid}
		"""
		s = list(intermLatentGrid.shape)
		gridH = s[2]
		gridW = s[3]

		useChunks = True
		chunkShape = self.chunkShape
		chunkStride = chunkShape // 2
		chunkOverlap = chunkStride // 2

		outSize = 2**(self.outRes - self.mergeLevel)
		if self.output is None or updateAll:
			self.output = np.zeros((3, outSize*gridH, outSize*gridW), np.uint8)
			start = (0, 0)
			end = (gridH, gridW)
			print('updating grid from {} to {}'.format(start, end))

		def roundUp(numToRound, multiple):
			return int((int((numToRound + multiple - 1) / multiple)) * multiple)

		with self.graphB.as_default():
			with self.sessionB.as_default():
				if useChunks: #process grid in chunks
					if self.GsB.input_shape[1:] != [s[1], chunkShape, chunkShape] :  # only update if input shape changed
						print('update network graph for different input shape {}x{}x{}'.format(s[1], chunkShape, chunkShape))
						self.GsB.update_latent_size(chunkShape, chunkShape)
					latentChunk = np.zeros((s[0], s[1], chunkShape, chunkShape))

					#rounding down to next multiple of chunkStride
					startY = (start[0] // chunkStride) * chunkStride
					startX = (start[1] // chunkStride) * chunkStride

					#rounding up to next multiple of chunkStride
					lastY = max(roundUp(end[0], chunkStride) - chunkShape, 0)
					lastX = max(roundUp(end[1], chunkStride) - chunkShape, 0)

					#print('lastY, lastX =', lastY, lastX)
					for y in range(startY, lastY + 1, chunkStride):
						for x in range(startX, lastX + 1, chunkStride):
							latentChunk.fill(0)

							chunkH = min(y + chunkShape, gridH) - y
							chunkW = min(x + chunkShape, gridW) - x

							#get image output for chunk
							latentChunk[:, :, :chunkH, :chunkW] = intermLatentGrid[:, :, y:y + chunkH, x:x + chunkW]
							try:
								_, chunkOutput = self.GsB.run(latentChunk, in_res=self.mergeLevel + 1, out_res=self.outRes, latent_size=[None, s[1], chunkShape, chunkShape], latent_depth=s[1], minibatch_size=32, num_gpus=1, out_mul=127.5, out_add=127.5, out_dtype=np.uint8)
								chunkOutput = np.squeeze(chunkOutput)
							except ValueError:
								print('latentChunk shape', latentChunk.shape)
								print('GsB input shape',  self.GsB.input_shape)

							#fix zero-padding issue by subtracting chunk overlap region from chunk borders, unless chunk is at image edge
							if x == 0:
								marginL = 0
							else:
								marginL = chunkOverlap
							if y == 0:
								marginT = 0
							else:
								marginT = chunkOverlap
							if x >= gridW - chunkShape:
								marginR = 0
							else:
								marginR = chunkOverlap
							if y >= gridH - chunkShape:
								marginB = 0
							else:
								marginB = chunkOverlap
							pasteRegion = chunkOutput[:, marginT*outSize:(chunkH-marginB)*outSize, marginL*outSize:(chunkW-marginR)*outSize]
							self.output[:, (y+marginT)*outSize: (y+chunkH-marginB)*outSize, (x+marginL)*outSize: (x+chunkW-marginR)*outSize] = pasteRegion
					return np.squeeze(self.output)
				else: #if chunking is disabled, process entire grid
					if s[1:] != self.GsB.input_shape[1:]:  # only update if input shape changed
						self.GsB.update_latent_size(s[2], s[3])
					_, self.output = self.GsB.run(intermLatentGrid, in_res=self.mergeLevel + 1, out_res=self.outRes, latent_size=[None, s[1], s[2], s[3]], latent_depth=s[1], minibatch_size=32, num_gpus=1, out_mul=127.5, out_add=127.5, out_dtype=np.uint8)
					return np.squeeze(self.output)

	def deadLeaves(self, height, width):
		"""
		initializing randomized latent field in larger coherent latent regions
		"""
		self.height = height
		self.width = width

		self.guidanceImage = None #randomized output doesn't have guidance image

		latentSize = self.latentSize
		pad = self.getLeftPadding()
		gridH = self.height * latentSize
		gridW = self.width * latentSize

		maxDim = int(min(gridW, gridH)*2/3) # TODO make more flexible

		self.latentList = np.zeros((gridH*gridW, self.latentDepth))
		self.intermLatentGrid = np.zeros((1, self.latentDepth, gridH, gridW))
		self.clusterGrid = np.zeros((self.height * self.latentSize, self.width * self.latentSize), dtype=np.uint8)
		occupancy = np.zeros((gridH, gridW), dtype=np.uint8)
		print('intermLatentGrid has shape', self.intermLatentGrid.shape)

		it = 0
		while np.any(occupancy == 0) or it > gridH*gridH*10: #second criterion is arbitrary emergency stopping condition for infinite loop - should not happen anymore anyway ;)
			it += 1
			randX = np.random.randint(gridW)
			randY = np.random.randint(gridH)

			# clamping is necessary because exponential distribution can be larger than beta
			randW = maxDim + 1
			randH = maxDim + 1
			while randW > maxDim or randW <= 0:
				randW = int(np.random.exponential(maxDim))
			while randH > maxDim or randH <= 0:
				randH = int(np.random.exponential(maxDim))

			endX = min(randX + randW, gridW)
			endY = min(randY + randH, gridH)
			actualW = endX - randX
			actualH = endY - randY

			print('{}x{} at ({}, {})'.format(actualW, actualH, randX, randY))
			randomIdx = np.random.randint(len(self.latentLookup))
			randomLatent = self.latentLookup[randomIdx]
			intermLatent = self.calculateIntermediateLatents(randomLatent)
			randomCluster = np.squeeze(self.clusterLookup[randomIdx])

			latentPatch = np.zeros((1, self.latentDepth, actualH, actualW))
			clusterPatch = np.zeros((actualH, actualW))
			for y in np.arange(actualH):
				for x in np.arange(actualW):
					yR = np.random.randint(intermLatent.shape[-2])
					xR = np.random.randint(intermLatent.shape[-1])
					latentPatch[:, :, y, x] = intermLatent[:, :, yR, xR]
					clusterPatch[y, x] = randomCluster
					self.latentList[y * gridW + x] = randomLatent

			#cut patch to correct dimensions
			self.intermLatentGrid[:, :, randY:endY, randX:endX] = latentPatch[:, :, :actualH, :actualW]
			self.clusterGrid[randY:endY, randX:endX] = clusterPatch[:actualH, :actualW]


			occupancy[randY:endY, randX:endX] = it

		if self.undoStack is None or self.undoStack.shape[1] != len(self.latentList):
			print('initialize undo stack...')
			self.initUndoStack()

		outputImage = self.calculateOutputImage(self.intermLatentGrid, updateAll=True)

		return outputImage, (self.intermLatentGrid.shape[2], self.intermLatentGrid.shape[3], self.latentSize, self.mergeLevel), self.undoCount

	def randomizeGrid(self, height, width):
		"""
		initializing randomized latent field one random latent per grid cell
		"""
		self.guidanceImage = None #randomized output doesn't have guidance image
		self.latentList = self.initLatentList(height, width, repeat=False)

		outputImage = self.getOutputFromLatents(self.latentList)
		return outputImage, (self.intermLatentGrid.shape[2], self.intermLatentGrid.shape[3], self.latentSize, self.mergeLevel), self.undoCount

	def getIntermediates(self):
		return self.intermLatents

	def getOutput(self):
		outputImage = self.getOutputFromLatents(self.latentList, updateUndoStack=False)
		return outputImage, (self.intermLatentGrid.shape[2], self.intermLatentGrid.shape[3], self.latentSize, self.mergeLevel)

	def getUnmergedOutput(self):
		return self.calculateUnmergedOutputImage(self.intermLatentGrid)

	def getClusterAt(self, sourceY, sourceX):
		print('cluster grid shape: {}'.format(self.clusterGrid.shape))
		return self.clusterGrid[sourceY, sourceX]

	def getClusterOutput(self):
		"""
		return a visualization of the cluster membership of each latent tile
		"""
		dominantColors = self.getDominantClusterColors()
		clusterOutput = np.zeros((self.height*self.latentSize, self.width*self.latentSize, 3), dtype=np.uint8)
		print(clusterOutput.shape)
		for y in range(self.height*self.latentSize):
			for x in range(self.width*self.latentSize):
				clusterOutput[y, x, :] = dominantColors[self.clusterGrid[y, x]]
		return clusterOutput

	def sampleFromCluster(self, latentCluster, numLatents=10, useCDF=True):
		"""
		draw a set of numLatents from cluster, upsample them and store them so the user can pick one particular one
		"""
		cluster = self.latentClusters[latentCluster]
		if useCDF:
			cdf = self.latentCDFs[latentCluster]
			randomIdxs = [ bisect(cdf, np.random.random()) for i in range(numLatents) ]
		else:
			randomIdxs = np.random.randint(len(cluster), size=numLatents)

		#looking up latent indices in latent bank in cluster first
		randLatents = np.asarray([ np.squeeze(self.latentLookup[ cluster[randomIdx] ]) for randomIdx in randomIdxs ])

		self.clusterSamples = randLatents

		randIntermLatents = self.calculateIntermediateLatents(randLatents)

		# decrust random latent
		pad = self.getLeftPadding()
		randIntermLatents = randIntermLatents[:, :, pad:pad + self.latentSize, pad:pad + self.latentSize]

		s = list(randIntermLatents.shape)
		with self.graphC.as_default():
			with self.sessionC.as_default():
				if [s[1], self.latentSize, self.latentSize] != self.GsC.input_shape[1:]:  # only update if input shape changed
					self.GsC.update_latent_size(self.latentSize, self.latentSize)
				_, outputs = self.GsC.run(randIntermLatents, in_res=self.mergeLevel + 1, out_res=self.outRes, latent_size=[None, s[1], self.latentSize, self.latentSize], latent_depth=s[1], minibatch_size=32, num_gpus=1, out_mul=127.5, out_add=127.5, out_dtype=np.uint8)

		return outputs

class Server(BaseManager):
	"""
	subclass BaseManager to communicate remotely
	"""
	def __init__(self, *args, ** kwargs):
		signal.signal(signal.SIGINT, ctrl_c)
		super().__init__(*args, **kwargs)

if __name__ == '__main__':
	port = 8080
	#get current IP for information regarding to connecting to manager
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.connect(("8.8.8.8", 80))
	ip = s.getsockname()[0]
	s.close()

	manager = TFManager()

	def ctrl_c(signal, frame):
		print('shutting down server...')
		manager.closeSessions()
		server.stop()

	server_process = Server(address=('', port), authkey=b'tilegan')
	server_process.register('sampleFromCluster', manager.sampleFromCluster)
	server_process.register('findDatasets', manager.findDatasets)
	server_process.register('initDataset', manager.initDataset)
	server_process.register('getLatentImages', manager.getLatentImages)
	server_process.register('getLatentAverages', manager.getLatentAverages)
	server_process.register('getDominantClusterColors', manager.getDominantClusterColors)
	server_process.register('getOutput', manager.getOutput)
	server_process.register('getUnmergedOutput', manager.getUnmergedOutput)
	server_process.register('getClusterOutput', manager.getClusterOutput)
	server_process.register('getClusterAt', manager.getClusterAt)
	server_process.register('getUpsampled', manager.getUpsampled)
	server_process.register('putLatent', manager.putLatent)
	server_process.register('perturbLatent', manager.perturbLatent)
	server_process.register('pasteLatents', manager.pasteLatents)
	server_process.register('saveLatents', manager.saveLatents)
	server_process.register('improveLatents', manager.MRFLatents)
	server_process.register('setMergeLevel', manager.setMergeLevel)
	server_process.register('randomizeGrid', manager.randomizeGrid)
	server_process.register('deadLeaves', manager.deadLeaves)
	server_process.register('undo', manager.undo)

	server = server_process.get_server()
	print('server is listening...')
	print('''
	 Connect application to TensorFlow manager at IP address {} and port {}.
	'''.format(ip, port))
	try:
		server.serve_forever()
	except (KeyboardInterrupt, SystemExit):
		print('khallas! My work here is done.')
