# 논구동 

## Algorithms 

|-|Algo| referenece  |
|:-:|:--|:-:|
|2022.04.21 🚀|intergrated Gradient, SmoothGrad,  |[TF](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients?hl=ko) [Paper](https://arxiv.org/abs/1703.01365)| 
|same | SmoothGrad: removing noise by adding noise | [paper](https://arxiv.org/abs/1706.03825)
|🚀|Gradient Class Activation Map |[github](https://github.com/jacobgil/pytorch-grad-cam)|
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
  - [ ] 

---
## Role 

돌아가면서 진행 

Method 1 
* Gradient Based (2~3주) 
  * CAM :  
  * IG : 
  * Smooth Gradient :
* 1 주차 


Method 2

- Analyze : 알고리즘 Psudo Code 분석 logic 설계 
- Coder (IMP) : 코드 구현

## 가장 단순한 데이터셋으로 검증 

- Tabular : IRIS 
- Image : MNIST 
- Timeseries : TBD
- NLP : TBD
- RL : CartPole 


## Study 

IG 
* model agnostic 
* base line 을 기준으로 변화량 측정 
