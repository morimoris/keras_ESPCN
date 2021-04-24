# keras_espcn
### detailed description(Japanese)

https://qiita.com/nekono_nekomori/items/08ec250ceb09a0004768

### Overview
I created Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network(ESPCN) using python and keras.

### Experiment environment
- OS : Windows 10
- CPU : AMD Ryzen 5 3500 6-Core Processor 8GB
- GPU : NVIDIA GeForce RTX 2060 SUPER

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

