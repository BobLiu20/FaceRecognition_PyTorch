# Face Recognition

### Models
[SphereFace with A-Softmax](https://arxiv.org/abs/1704.08063)

### Training
```
python training.py --input_paths=/your/data/path/10000_caffe_format.lst --working_root=/your/path/sphereface_pytorch --max_epoch=100 --img_size=112 --feature_dim=512 --label_num=10000 --process_num=15 --learning_rate=0.1 --model=SphereFaceNet --model_params='{"lamb_iter":0,"lamb_base":1000,"lamb_gamma":0.00001,"lamb_power":1,"lamb_min":500, "layer_type": "20layer"}' --try=0 --gpu_device=0,1,2,3 --batch_size=32
```

### Testing
TODO

### References
[sphereface](https://github.com/wy1iu/sphereface)
[sphereface](https://github.com/clcarwin/sphereface_pytorch)

