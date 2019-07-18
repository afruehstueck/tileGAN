## TileGAN: Synthesis of Large-Scale Non-Homogeneous Textures
![TileGAN](https://github.com/afruehstueck/tileGAN/blob/master/doc/tilegan_results.jpg)
We tackle the problem of texture synthesis in the setting where many input
images are given and a large-scale output is required. We build on recent
generative adversarial networks and propose two extensions in this paper.
First, we propose an algorithm to combine outputs of GANs trained on
a smaller resolution to produce a large-scale plausible texture map with
virtually no boundary artifacts. Second, we propose a user interface to
enable artistic control. Our quantitative and qualitative results showcase the
generation of synthesized high-resolution maps consisting of up to hundreds
of megapixels as a case in point.

## Code
The TileGAN application consists of two independent processes, the server and the client. Both can be run locally on your machine or you can choose to run the server on a remote location, depending on your hardware setup. All network operations are performed by the server process, which sends the result to the client for displaying.
### Download our pre-trained networks
 [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/The_Scream.jpg/471px-The_Scream.jpg" height="128"><img src="https://raw.githubusercontent.com/afruehstueck/tileGAN/master/doc/munch.jpg" height="128">](https://drive.google.com/file/d/1Cmgblx8zKiv-W1IHXqi7By4fAnvBWsyR)
 [<img src="https://upload.wikimedia.org/wikipedia/en/0/05/Un_dimanche_apr%C3%A8s-midi_%C3%A0_l%27%C3%8Ele_de_la_Grande_Jatte_crop.jpg" height="128"><img src="https://raw.githubusercontent.com/afruehstueck/tileGAN/master/doc/seurat.jpg" height="128">](https://drive.google.com/file/d/1PQwAOPetysY6VKNWRW8EqNPPtfSrnf2x)
 [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/606px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg" height="128"><img src="https://raw.githubusercontent.com/afruehstueck/tileGAN/master/doc/vangogh.jpg" height="128">](https://drive.google.com/file/d/1nJyCRiz6XmJXI71DEVeXWq49EvTbII0S)
* Download the network(s) to the location of your server (this can be your local machine or a remote server)
 - [<img src="https://raw.githubusercontent.com/afruehstueck/tileGAN/master/doc/munch.jpg" height="32">](https://drive.google.com/file/d/1Cmgblx8zKiv-W1IHXqi7By4fAnvBWsyR) [The Scream (Edvard Munch)](https://drive.google.com/file/d/1Cmgblx8zKiv-W1IHXqi7By4fAnvBWsyR)
 -  [<img src="https://raw.githubusercontent.com/afruehstueck/tileGAN/master/doc/seurat.jpg" height="32">](https://drive.google.com/file/d/1PQwAOPetysY6VKNWRW8EqNPPtfSrnf2x) [Un dimanche après-midi (Georges Seurat)](https://drive.google.com/file/d/1PQwAOPetysY6VKNWRW8EqNPPtfSrnf2x)
 - [<img src="https://raw.githubusercontent.com/afruehstueck/tileGAN/master/doc/vangogh.jpg" height="32">](https://drive.google.com/file/d/1nJyCRiz6XmJXI71DEVeXWq49EvTbII0S) [Starry Night (Vincent Van Gogh)](https://drive.google.com/file/d/1nJyCRiz6XmJXI71DEVeXWq49EvTbII0S)
 * Extract the `.zip` to `./data`. There should be a separate folder for each dataset in `./data` (e.g. `./data/vangogh`) containing a `*_network.pkl`, a `*_descriptors.hdf5`, a `*_clusters.hdf5` and a `*_kmeans.joblib` file.
 
### Setup server
* Install requirements from `requirements-pip.txt`
* Install [hnswlib](https://github.com/nmslib/hnswlib):
    ```
    git clone https://github.com/nmslib/hnswlib.git
  cd hnswlib/python_bindings
  python setup.py install
  ``` 
 * run `python tileGAN_server.py`
 * the server process will start, then tell you the IP to connect to.
 
### Setup client
 * Install Qt for Python. The easiest way to do this is using [conda](https://www.anaconda.com/distribution/#download-section): `conda install -c conda-forge qt pyside2`
 * Install requirements from `requirements-pip.txt`
 * Run `python tileGAN_client.py XX.XX.XX.XX` (insert the IP from the server). If you're running the server on the same machine as the application, you can omit the IP address or use 'localhost'.

### Using your own data/network (optional)
 * Train a network on own your data using [Progressive Growing of GANs](https://github.com/tkarras/progressive_growing_of_gans)
 * run `create_dataset path_to_pkl num_latents t_size num_clusters network_name` (expected to take between 10 and 60 minutes depending on the specified sample size) 
    - `path_to_pkl` the path to the trained network pickle of your ProGAN network
    - `num_latents` the size of the database entries (a good number would be 50K to 300K)
    - `t_size` the size of the output descriptors (an even number somewhere around 12 and 24)
    - `num_clusters` the number of clusters (approx. 8-16)
    - `network_name` the name you want to assign your network
 * run server and client and load network in UI from the drop down menu. First time the network is loaded, an ANN index is created (expected to take <5mins depending on sample size)

### Using our application
![TileGAN Tutorial](https://raw.githubusercontent.com/afruehstueck/tileGAN/master/tileGAN_firstSteps.jpg)

## Video
Watch our video on Youtube:
[<img src="https://github.com/afruehstueck/tileGAN/blob/master/doc/video_link.jpg">](https://www.youtube.com/watch?v=ye_HZOdW7kg)
## Results
Some of our results can be viewed interactively on easyzoom:

[<img src="https://easyzoom.blob.core.windows.net/tiled/d6aaea2c-78a0-4f3b-9910-a92f1cae65c2/d6aaea2c-78a0-4f3b-9910-a92f1cae65c2_272.jpg">](https://www.easyzoom.com/imageaccess/2c874aa0cfc9478eaff289c0ad41cbb7)
[<img src="https://easyzoom.blob.core.windows.net/tiled/ec0082ce-5ead-4c74-aadc-eba4be3487d1/ec0082ce-5ead-4c74-aadc-eba4be3487d1_272.jpg">](https://www.easyzoom.com/imageaccess/62409020a0334402b2c91d9f6aa63459)
[<img src="https://easyzoom.blob.core.windows.net/tiled/c2561974-2010-48da-ac87-1216f4eb2e7a/c2561974-2010-48da-ac87-1216f4eb2e7a_272.jpg">](https://www.easyzoom.com/imageaccess/d16837356655462bb034a6e2c6c209d8)

## Paper
available on [arXiv](https://arxiv.org/abs/1904.12795) or [ACM](https://dl.acm.org/citation.cfm?id=3322993)

## Authors
[Anna Frühstück](http://afruehstueck.github.io), [Ibraheem Alhashim](http://ialhashim.github.io), [Peter Wonka](http://peterwonka.net)

## Citation
If you use this code for your research, please cite our paper:
```
@article{Fruehstueck2019TileGAN,
  title      = {{TileGAN}: Synthesis of Large-Scale Non-Homogeneous Textures},
  author     = {Fr\"{u}hst\"{u}ck, Anna and Alhashim, Ibraheem and Wonka, Peter},
  journal    = {ACM Transactions on Graphics (Proc. SIGGRAPH) },
  issue_date = {July 2019},
  volume     = {38},
  number     = {4},
  year       = {2019}
}
```

## Acknowledgements
Our project is based on [ProGAN](https://github.com/tkarras/progressive_growing_of_gans). We'd like to thank Tero Karras et al. for their great work and for making their code available.
