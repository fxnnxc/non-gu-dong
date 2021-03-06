# λΌκ΅¬λ 

## Algorithms 

|-|Algo| referenece  |
|:-:|:--|:-:|
|2022.04.21 π|intergrated Gradient, SmoothGrad,  |[TF](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients?hl=ko) [Paper](https://arxiv.org/abs/1703.01365)| 
|same | SmoothGrad: removing noise by adding noise | [paper](https://arxiv.org/abs/1706.03825)
|π|Gradient Class Activation Map |[github](https://github.com/jacobgil/pytorch-grad-cam)|
| |Genrealized Intersection over Union | [Stanford](https://giou.stanford.edu/)|
| |Gabor Filter | [Wiki](https://en.wikipedia.org/wiki/Gabor_filter)
| | Gernerative Model ! | [post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)|
| | Neural ODE ! | [post]() | 


- [ ] Sampling
- [ ] Probability Distance  
  - [ ] Wassertein
  - [ ] MMD 
  - [ ] Total Variance
  - [ ] Kullback Leibler  
- [ ] Tree 
  - [ ] XGBoosting 
  - [ ] CatBoost 
  - [ ] LightGBM
- [ ] Shapley 
- [ ] Layer-wise Relevance Propagation 
  - [ ] Spray (Spectral Clustering)
- [ ] Computer vision 
  - [ ] non-maximum suppression
  - [ ] IoU / GIoU 
  - [ ] segementation 
  - [ ] image filter 
  - [ ] edge detection 
  - [ ] Phase Stretch Transform (PST)  
  - [ ] Gabor filter
- [ ] NLP 
  - [ ] Beam Search 
  - [ ] TextRank
  - [ ] Tokenization
- [ ] Matrix Dimension Based 
  - [ ] QR Factorization 
  - [ ] PCA 
  - [ ] incremental PCA
  - [ ] Power Method 
  - [ ] sinkhorn 
  - [ ] Iterative proportional fitting
- [ ] XAI
  - [x] IG
  - [ ] SmoothGrad : 
  - [ ] Vanilla Grad :
  - [ ] LRP
    - [ ] Gamma Rule
    - [ ] Epsilon 
    - [ ] Z_plus rule 
  - [ ] FullGrad  
  - [ ] Grad-CAM 
  - [ ] CAM
- [ ] Generative Model 
  - [ ] GAN
  - [ ] VAE 
  - [ ] AE 
  - [ ] Diffusion Model 
  - [ ] Flow-based Model 



---
## Role 

λμκ°λ©΄μ μ§ν 

Method 1 
* Gradient Based (2~3μ£Ό) 
  * CAM :  
  * IG : 
  * Smooth Gradient :
* 1 μ£Όμ°¨ 


Method 2

- Analyze : μκ³ λ¦¬μ¦ Psudo Code λΆμ logic μ€κ³ 
- Coder (IMP) : μ½λ κ΅¬ν

## κ°μ₯ λ¨μν λ°μ΄ν°μμΌλ‘ κ²μ¦ 

- Tabular : IRIS 
- Image : MNIST / CIFA 10
- Timeseries : TBD
- NLP : TBD
- RL : CartPole 


## Study 

### 2022.04.21
IG / Smooth / Vanila  
* model agnostic 
* base line μ κΈ°μ€μΌλ‘ λ³νλ μΈ‘μ  

### 2022.04.28
CAM / GradCAM / FullCAM
