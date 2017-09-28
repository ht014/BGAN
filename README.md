# BGAN

To run this codes, you should do follow things:<p>
1. extract resnet feature and then run create_S.py to construct similarity matrix.<br>  You can download cifar-10.mat training data and  cifar_KNN.npz from <a href="http://pan.baidu.com/s/1geUCy0F"> here </a> 
2. download vgg19 pretrained model on ImageNet based tensorflow.
3. run this command python BGAN.py 32 to train network and then generate 32 bit codes
4. after training done, you can run evalutate.py to calculate MAP. HIT: you need to change some paths in evalutate.py.
