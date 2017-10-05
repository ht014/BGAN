# BGAN

To run this codes, you should do follow things:<p>
1. extract resnet feature and then run create_S.py to construct similarity matrix.<br>  You can download cifar-10.mat <a href="https://drive.google.com/open?id=0Bzg9TvY-s7y2Zy1CQklaTTJQdUU"> (here) </a>training data and  cifar_KNN.npz from <a href="https://drive.google.com/open?id=0Bzg9TvY-s7y2WFFlc3F0T2RkalE"> (here) </a> 
2. download vgg19 pretrained model on ImageNet based tensorflow <a href='https://drive.google.com/open?id=0Bzg9TvY-s7y2UE12NVR6MEpxNUE'> here </a>.
3. run this command 'python BGAN.py 32' to train network and then generate 32 bit codes
4. after training done, you can run evalutate.py to calculate MAP. HIT: you need to change some paths in evalutate.py.
