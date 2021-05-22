# keras_espcn
### Detailed description(Japanese)

https://qiita.com/morimoris/items/08ec250ceb09a0004768

### Overview
I created Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network(ESPCN) using python and keras.

### Experiment environment

- PC environment
  - `CPU` : AMD Ryzen 5 3500 6-Core Processor
  - `メモリ数` : 40GB
  - `GPU` : NVIDIA GeForce RTX 2060 SUPER
  - `OS` : Windows 10
  
- Library environment
  - `python` : 3.7.9
  - `tensorflow-gpu` : 2.4.1
  - `keras` : 2.4.3
  - `opencv-python` : 4.4.0.43

### How to use
1. Create new folders which are `./train_data` and `./test_data`
   
   Storage train data in `./train_data` and test data in `./test_data`
2. Learning
```
main.py --mode srcnn
```
3. Evaluate
```
main.py --mode evaluate
```
### Result example
#### High Image
![High Image](result_epo_1000/high_0.jpg)

#### Low Image 
![Low Image](result_epo_1000/low_0.jpg)
#### Pred Image(three enlargements) PSNR:31.88dB
![Pred Image](result_epo_1000/pred_0.jpg)

