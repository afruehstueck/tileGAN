## TileGAN: Synthesis of Large-Scale Non-Homogeneous Textures
![TileGAN](https://github.com/afruehstueck/tileGAN/blob/master/doc/tilegan_results.jpg)

:bulb: *We will make the code of our paper **"TileGAN: Synthesis of Large-Scale Non-Homogeneous Textures"** (will be presented at SIGGRAPH 2019) available here soon.*

### Abstract
We tackle the problem of texture synthesis in the setting where many input
images are given and a large-scale output is required. We build on recent
generative adversarial networks and propose two extensions in this paper.
First, we propose an algorithm to combine outputs of GANs trained on
a smaller resolution to produce a large-scale plausible texture map with
virtually no boundary artifacts. Second, we propose a user interface to
enable artistic control. Our quantitative and qualitative results showcase the
generation of synthesized high-resolution maps consisting of up to hundreds
of megapixels as a case in point.


### Authors
[Anna Frühstück](http://afruehstueck.github.io), [Ibraheem Alhashim](http://ialhashim.github.io), [Peter Wonka](http://peterwonka.net)
### Citation
If you use this code for your research, please cite our paper:
```
@article{Fruehstueck2019TileGAN,
  title = {{TileGAN}: Synthesis of Large-Scale Non-Homogeneous Textures},
  author = {Fr\"{u}hst\"{u}ck, Anna and Alhashim, Ibraheem and Wonka, Peter},
  journal = {ACM Transactions on Graphics (Proc. SIGGRAPH) },
  issue_date = {July 2019},
  volume = {38},
  number = {4},
  month = jul,
  pages = {0},
  year = {2019}
}
```

### Acknowledgements
Our project is based on [ProGAN](https://github.com/tkarras/progressive_growing_of_gans). We'd like to thank Tero Karras et al. for their great work and for making their code available.
