# Face Recognition

### Models
1. **SphereFace with A-Softmax**
   * Review paper: [link](https://arxiv.org/abs/1704.08063)

### Environment
 * PyTorch 0.3.0.post4
 * Python 2.7.13

### Training

In this project, there are two different parallel method have been implemented.   
1. **Data Parallel**   
A good choice for this way when your class number is less than 300000(Maybe?).   
   Relate code:   
   * training/training.py   
   * models/sphere_face_net.py  

   Start training with this cmd:   
   ```
   python training.py --input_paths=/your/data/path/10000_caffe_format.lst --working_root=/your/path/sphereface_pytorch --max_epoch=100 --img_size=112 --feature_dim=512 --label_num=10000 --process_num=15 --learning_rate=0.1 --model=SphereFaceNet --model_params='{"lamb_iter":0,"lamb_base":1000,"lamb_gamma":0.00001,"lamb_power":1,"lamb_min":500, "layer_type": "20layer"}' --try=0 --gpu_device=0,1,2,3 --batch_size=128
   ```

2. Model Parallel   
We can not fit full model into one GPU when class number is huge in last margin fc layer.   
So? The model parallel is a good idea!!!   
   Relate code:   
   * training/training_model_parallel.py   
   * models/sphere_face_net_parallel.py   

   Start training with this cmd:   
   ```
   python training_model_parallel.py --input_paths=/your/data/path/1000000_caffe_format.lst --working_root=/your/path/sphereface_pytorch --max_epoch=100 --img_size=112 --feature_dim=512 --label_num=1000000 --process_num=15 --learning_rate=0.1 --model=SphereFaceNetParallel --model_params='{"lamb_iter":0,"lamb_base":1000,"lamb_gamma":0.00001,"lamb_power":1,"lamb_min":500, "layer_type": "20layer"}' --try=0 --gpu_device=0,1,2,3 --batch_size=128
   ```

### Testing & Benchmark
TODO

### References
[Pytorch](http://pytorch.org/docs/0.3.1/index.html)   
[sphereface](https://github.com/wy1iu/sphereface)   
[sphereface](https://github.com/clcarwin/sphereface_pytorch)   

