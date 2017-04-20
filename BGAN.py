import matplotlib
matplotlib.use('pdf')
from queue_context import queue_context
import tensorflow as tf
import numpy as np
from generator import Vgg19
from discriminator import discriminator
import os
import utils
import scipy
import scipy.io as sio
from deconv import deconv2d
from optimizer import optim
import tqdm
import scipy
import scipy.io as sio
import prettytensor as pt
import matplotlib.pyplot as plt
from deconv import deconv2d
import IPython.display
import math
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed

save_dir_fig ='reconsts'
if False==os.path.exists(save_dir_fig):
    os.mkdir(save_dir_fig)

def saveB(name_B):
    a = sio.loadmat('cifar-10.mat')
    dataset = a['data_set']
    test = a['test_data']
    images_ = []
    for i in tqdm.tqdm(range(59000)):
        t = dataset[:, :, :, i]
        image = scipy.misc.imresize(t, [224, 224])
        images_.append(image)
    images_ = np.array(images_)
    test_images = []
    for i in tqdm.tqdm(range(1000)):
        t = test[:, :, :, i]
        image = scipy.misc.imresize(t, [224, 224])
        test_images.append(image)
    test_images = np.array(test_images)
    for i in xrange(0, len(images_), batch_size):
        all = images_[i:i + batch_size]
        feature = sess.run([z_x_mean], \
                           feed_dict={all_input224:all,beta_nima:[-2] ,train_model:False})
        if i == 0:
            B = feature[0]
        else:
            B = np.concatenate((B, feature[0]), axis=0)

    for i in xrange(0, len(test_images), batch_size):
        all = test_images[i:i + batch_size]
        feature =  sess.run([z_x_mean], \
                            feed_dict={all_input224:all,beta_nima:[-2] ,train_model:False})
        if i == 0:
            B_ = feature[0]
        else:
            B_ = np.concatenate((B_, feature[0]), axis=0)
    print B.shape,B_.shape
    np.savez(name_B+'.npz', dataset=B, test=B_)


def data_iterator():
    while True:
        idxs = np.arange(0, len(img224))
        np.random.shuffle(idxs)
        for batch_idx in range(0, len(img224), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            images_batch = img224[cur_idxs]
            if len(images_batch) < batch_size:
                break
            images_batch = images_batch.astype("float32")
            yield images_batch ,cur_idxs

def generator(Z):
    return (pt.wrap(Z).
            fully_connected(8 * 8 * 256).
            reshape([batch_size, 8, 8, 256]).
            deconv2d(5, 256, stride=2).
            deconv2d(5, 128, stride=2).
            deconv2d(5, 32, stride=2).
            deconv2d(1, 3, stride=1, activation_fn=tf.sigmoid)
            )


def discriminator(D_I):

    descrim_conv = (pt.wrap(D_I).
                    reshape([batch_size, 64, 64, 3]).
                    conv2d(5, 32, stride=1).
                    conv2d(5, 128, stride=2).
                    conv2d(5, 256, stride=2).
                    conv2d(5, 256, stride=2).
                    flatten()
                    )
    lth_layer = descrim_conv.fully_connected(1024, activation_fn=tf.nn.elu)  # this is the lth layer
    D = lth_layer.fully_connected(1, activation_fn=tf.nn.sigmoid)  # this is the actual discrimination
    return D, lth_layer

def inference(x224,x64):
    z_p = tf.random_normal((batch_size, hidden_size), 0, 1)
    eps = tf.random_normal((batch_size, hidden_size), 0, 1)  # normal dist for VAE
    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with tf.variable_scope("enc"):
                vgg_net = Vgg19('./vgg19.npy', codelen=hidden_size)
                vgg_net.build(x224, beta_nima, train_model)
                z_x_mean = vgg_net.fc8
                z_x_log_sigma_sq = vgg_net.fc10

        with tf.variable_scope("gen"):
            z_x = tf.add(z_x_mean,
                         tf.multiply(tf.sqrt(tf.exp(z_x_log_sigma_sq)), eps))  # grab our actual z
            x_tilde = generator(z_x)
            # x_tilde = generator(z_x_mean)
        with tf.variable_scope("dis"):
            _, l_x_tilde = discriminator(x_tilde)
        with tf.variable_scope("gen", reuse=True):
            x_p = generator(z_p)
        with tf.variable_scope("dis", reuse=True):
            d_x, l_x = discriminator(x64)
        with tf.variable_scope("dis", reuse=True):
            d_x_p, _ = discriminator(x_p)
        return z_x_mean,z_x_log_sigma_sq,x_tilde, l_x_tilde, z_x,x_p, d_x, l_x, d_x_p, z_p



def loss(x64, x_tilde, z_x_log_sigma_sq1, z_x_meanx1, d_x, d_x_p, l_x, l_x_tilde,ss_ ):

    SSE_loss = tf.reduce_mean(tf.square(x64 - x_tilde))


    pair_loss=tf.reduce_mean(tf.square(tf.matmul(z_x_meanx1, tf.transpose(z_x_meanx1))- ss_)) +\
              tf.reduce_mean(tf.square(z_x_meanx1 - tf.sign(z_x_meanx1)))

    KL_loss = tf.reduce_sum(-0.5 * tf.reduce_sum(1 + tf.clip_by_value(z_x_log_sigma_sq1, -10.0, 10.0)
                                                 - tf.square(tf.clip_by_value(z_x_meanx1, -10.0, 10.0))
                                                 - tf.exp(tf.clip_by_value(z_x_log_sigma_sq1, -10.0, 10.0)),
                                                 1)) / 64/64/3

    D_loss = tf.reduce_mean(-1. * (tf.log(tf.clip_by_value(d_x, 1e-5, 1.0)) +
                                   tf.log(tf.clip_by_value(1.0 - d_x_p, 1e-5, 1.0))))
    G_loss = tf.reduce_mean(-1. * (tf.log(tf.clip_by_value(d_x_p, 1e-5, 1.0))))
    LL_loss = tf.reduce_sum(tf.square(l_x - l_x_tilde)) / 64/64/3
    return SSE_loss, KL_loss, D_loss, G_loss, LL_loss,pair_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

batch_size = 25
graph = tf.Graph()
import sys
hidden_size = int(sys.argv[1])

with graph.as_default():
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    lr_D = tf.placeholder(tf.float32, shape=[])
    lr_G = tf.placeholder(tf.float32, shape=[])
    lr_E = tf.placeholder(tf.float32, shape=[])
    opt_D = tf.train.AdamOptimizer(lr_D, epsilon=1.0)
    opt_G = tf.train.AdamOptimizer(lr_G, epsilon=1.0)
    opt_E = tf.train.AdamOptimizer(lr_E, epsilon=1.0)

with graph.as_default():
    tower_grads_e = []
    tower_grads_g = []
    tower_grads_d = []
    all_input224 = tf.placeholder(tf.float32, [batch_size , 224 ,224,3])
    all_input64 = tf.placeholder(tf.float32, [batch_size, 64, 64, 3])
    LL_param = tf.placeholder(tf.float32,shape=[])
    beta_nima = tf.placeholder(tf.float32,[1])
    G_param = tf.placeholder(tf.float32,shape=[])
    P_param = tf.placeholder(tf.float32,shape=[])
    train_model = tf.placeholder(tf.bool)
    # s_s = tf.placeholder(tf.float32, [batch_size,num_examples])
    s_s = tf.placeholder(tf.float32, [batch_size, batch_size])
    #U_T = tf.placeholder(tf.float32,[num_examples,hidden_size])

    with tf.device('/gpu:0'):
        with tf.name_scope('Tower_0') as scope:
            z_x_mean,z_x_log_sigma_sq,x_tilde, l_x_tilde, z_x, x_p, d_x, l_x, d_x_p, z_p = \
                inference(all_input224,all_input64)
            SSE_loss, KL_loss, D_loss, G_loss, LL_loss, pair_loss = loss(all_input64, x_tilde, z_x_log_sigma_sq, \
                                                                         z_x_mean, d_x, d_x_p, l_x,
                                                                         l_x_tilde, s_s)

            params = tf.trainable_variables()
            params = tf.trainable_variables()
            E_params = [i for i in params if 'enc' in i.name]
            G_params = [i for i in params if 'gen' in i.name]
            D_params = [i for i in params if 'dis' in i.name]

            grads_e = opt_E.compute_gradients(KL_loss+LL_loss+P_param*pair_loss, var_list=E_params)#with KL_loss,you can discard it.
            grads_g = opt_G.compute_gradients(LL_loss+G_loss, var_list=G_params)
            grads_d = opt_D.compute_gradients(D_loss, var_list=D_params)

            tower_grads_e.append(grads_e)
            tower_grads_g.append(grads_g)
            tower_grads_d.append(grads_d)


with graph.as_default():
    # Average the gradients
    grads_e = average_gradients(tower_grads_e)
    grads_g = average_gradients(tower_grads_g)
    grads_d = average_gradients(tower_grads_d)

    # apply the gradients with our optimizers
    train_E = opt_E.apply_gradients(grads_e, global_step=global_step)
    train_G = opt_G.apply_gradients(grads_g, global_step=global_step)
    train_D = opt_D.apply_gradients(grads_d, global_step=global_step)


with graph.as_default():
    saver = tf.train.Saver()  # initialize network saver
    sess = tf.InteractiveSession(graph=graph,config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
    sess = tf.InteractiveSession(graph=graph,config=config)
    sess.run(tf.global_variables_initializer())


def sigmoid(x, shift, mult):
    return 1 / (1 + math.exp(-(x + shift) * mult))

betas=[-2,-5,-10,-15,-20]
S = sio.loadmat('../S_K1_20_K2_30.mat')['S']  # similarity matrix
dataset1 = sio.loadmat('cifar-10.mat')['train_data']  #cifar-10 data
img224 = []
img64 = []
for i in tqdm.tqdm(range(10000)):
    t = dataset1[:, :, :, i]
    image1 = scipy.misc.imresize(t, [224, 224])
    img224.append(image1)
    img2 = scipy.misc.imresize(t, [64, 64]).astype(np.float32)
    img64.append(img2/255.0)

img224 = np.array(img224)
img64 = np.array(img64)
num_examples = len(img64)
total_batch = int(np.floor(num_examples / batch_size ))
epoch = 0
d_real = 0
d_fake = 0
cur_epoch = 0
num_epochs =  101
e_learning_rate = 1e-3
g_learning_rate = 1e-3
d_learning_rate = 1e-3
globa_beta_indx = 0
while epoch < num_epochs:
    iter_ = data_iterator()
    for i in  range(total_batch):
        cur_epoch += 1.0
        e_current_lr = e_learning_rate*1.0#/betas[globa_beta_indx] #* sigmoid(np.mean(d_real), -.5, 15)
        g_current_lr = g_learning_rate #* sigmoid(np.mean(d_real), -.5, 15)
        d_current_lr = d_learning_rate #* sigmoid(np.mean(d_fake), -.5, 15)
        next_batches224 ,indx3= iter_.next()
        next_batches64 = img64[indx3]
        ss_ = S[indx3,:][:,indx3]
        z_mn, _, _, _, D_err, KL_err,G_err, SSE_err, LL_err, PP_err, d_fake, d_real = sess.run(
            [z_x_mean,
             train_E, train_G, train_D,
             D_loss,KL_loss, G_loss,SSE_loss,
             LL_loss, pair_loss,
             d_x_p, d_x,
             ],
            {
                lr_E: e_current_lr,
                lr_G: e_current_lr,
                lr_D: e_current_lr,
                all_input224: next_batches224,
                all_input64: next_batches64,
                G_param: 0.1,
                LL_param: 0.1,
                beta_nima:[betas[globa_beta_indx]],
                P_param: 1.0,
                s_s: ss_,
                train_model: True

            }
            )

        print "epoch:{0},all_loss:{1},SSE_loss:{2}".\
            format(cur_epoch/total_batch,PP_err+KL_err+LL_err+D_err+G_err,SSE_err)

    #IPython.display.clear_output()
    #plot_network_output(str(epoch).zfill(5))
    epoch += 1
 saver.save(sess, ''.join(['models/cifar_pair_KL_',str(hidden_size),'_', str(epoch).zfill(4), '.tfmod']))
