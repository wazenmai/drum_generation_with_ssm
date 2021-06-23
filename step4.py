import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import librosa, IPython, pickle, datetime, time, os, sys, copy, glob
from time import gmtime, strftime
import numpy as np
import tensorflow as tf
from tf_ops import *
from tf_util import *
import matplotlib.pyplot as plt
# %matplotlib inline

tf.compat.v1.disable_eager_execution()

# show version info
print ("[info] Current Time:     " + datetime.datetime.now().strftime('%Y/%m/%d  %H:%M:%S'))
print ("[info] Python Version:   " + sys.version.split('\n')[0].split(' ')[0])
print ("[info] Working Dir:      " + os.getcwd()+'/')
print ("[info] Tensorflow:       " + tf.__version__)

# enable gpu usage constraint here
limited_gpu_usage = 1; occupied_gpu_dev = 0;

# if gpu usage is constraint, limit certain gpu for use
if (limited_gpu_usage == 1):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"                       # list GPU sequence by PCI bus GPU ID                   
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(occupied_gpu_dev)   

    # check available GPU
    from tensorflow.python.client import device_lib
    for x in range(1, len(device_lib.list_local_devices())):
        print ("[info] GPU device:       " + device_lib.list_local_devices()[x].physical_device_desc[17:])


### Reload song/bar index code
with open('./pre_processed_data/abs_bar_idx_str_list.pkl', 'rb') as pkl_file:        
    abs_bar_idx_str_list = pickle.load(pkl_file)
    
print ('[info] List of [song/bar] data is loaded.')
print ('[info] Total bars: {}'.format(len(abs_bar_idx_str_list)))
print ('[info] First 5 bar code: {}'.format(abs_bar_idx_str_list[:5]))
print ('[info] Last  5 bar code: {}'.format(abs_bar_idx_str_list[-5:]))

### Define basic CQT read function
read_pooled_cqt_flist = np.sort(glob.glob('./pre_processed_data/cqt_pooled_data/*.pkl', recursive=True)).tolist()
print ('[info] Total # of CQT files: {}'.format(len(read_pooled_cqt_flist)))

def get_bar_cqt_data(abd_bar_idx_str):
    
    song_idx = int(abd_bar_idx_str.split('_')[0])
    bar_idx = int(abd_bar_idx_str.split('_')[1])
    
    file_name = read_pooled_cqt_flist[song_idx]
    with open(file_name, 'rb') as pkl_file:
        pooled_cqt_data = pickle.load(pkl_file)  
        
    bar_cqt_data = pooled_cqt_data[bar_idx]
    
    return(bar_cqt_data)
    
print ('[info] CQT reading function defined.')


### Define function to load drum data
file_name = './pre_processed_data/cdsed_drum_bar_list_28_46/song_drum_bar_list_46.pkl'   
with open(file_name, 'rb') as pkl_file:
    song_drum_bar_list_46 = pickle.load(pkl_file)

song_idx = 10; bar_idx = 30;
print('[info] Total # of drum tracks: {}'.format(len(song_drum_bar_list_46)))
print('[info] Single-bar drum data shape: {}'.format(song_drum_bar_list_46[song_idx][bar_idx].shape))

### Set note complexity parameter list [+0, +3, +6, +12, +20]
bar_add_note_num_list = [0, 3, 6, 12, 20]

### Define function to get data by index code
# read bar selection data
with open('./pre_processed_data/vaegan_bar_selection_index_list.pkl', 'rb') as pkl_file:
    bar_selection_list = pickle.load(pkl_file)

# read song attribute
with open('./pre_processed_data/all_song_attribute.pkl', 'rb') as pkl_file:
    song_attribute_data_list = pickle.load(pkl_file)    


# define python read function
def read_pkl_function(index_code_in):
    
    # convert data into correct type
    if type(index_code_in)!=str:
        index_code_in = index_code_in.numpy().decode("utf-8")
        
    # extract information from index code
    song_idx_str = index_code_in.split('_')[0]
    song_idx_int = int(song_idx_str)
    bar_idx_str = index_code_in.split('_')[1]
    bar_idx_int = int(bar_idx_str)
    
    # set parameter to get relative bars
    get_n_rtv_bars = 7
    rtv_bar_index = np.round(bar_selection_list[int(song_idx_str)][int(bar_idx_str), 0, 0:get_n_rtv_bars]).astype(int)
    rtv_bar_ratio = np.hstack([1.0, bar_selection_list[int(song_idx_str)][int(bar_idx_str), 1, 0:get_n_rtv_bars]]).astype(np.float32)
    
    # collect 8-bars CQT data
    cqt_data_rtv_bar_list = []
    
    for cqt_bar_idx in range(0, 8):
        
        # get current bar data
        if cqt_bar_idx==0:
            cqt_data_rtv_bar_list.append(get_bar_cqt_data(index_code_in))
            
        # get 7-relative bars data
        else:
            index_code_tmp = song_idx_str + '_' + '{:0>3}'.format(rtv_bar_index[cqt_bar_idx-1])
            cqt_data_rtv_bar_list.append(get_bar_cqt_data(index_code_tmp))
    
    # process CQT data
    cqt_data_fname_rtv_mix = np.concatenate([cqt_data_rtv_bar_list[0][:,:,np.newaxis],
                                             cqt_data_rtv_bar_list[1][:,:,np.newaxis],
                                             cqt_data_rtv_bar_list[2][:,:,np.newaxis],
                                             cqt_data_rtv_bar_list[3][:,:,np.newaxis],
                                             cqt_data_rtv_bar_list[4][:,:,np.newaxis],
                                             cqt_data_rtv_bar_list[5][:,:,np.newaxis],
                                             cqt_data_rtv_bar_list[6][:,:,np.newaxis],
                                             cqt_data_rtv_bar_list[7][:,:,np.newaxis]], axis=-1).astype(np.float32)
            
    # reload original MIDI drum data for reference
    drum_bar_data_reload = song_drum_bar_list_46[song_idx_int][bar_idx_int][:,:,np.newaxis].astype(np.float32)
    

    # process song attributes    
    attribute_data_list = song_attribute_data_list[song_idx_int]
    tempo_norm_v_oh =     attribute_data_list[1].astype(np.float32)
    style_tag_array_oh =  attribute_data_list[2].astype(np.float32)
    song_progress_oh =    attribute_data_list[3][int(bar_idx_str),:].astype(np.float32)
    n_note_in_bar =       np.array([attribute_data_list[4][int(bar_idx_str)]]).astype(np.float32)
    
    
    # send back data
    return (cqt_data_fname_rtv_mix,  \
            rtv_bar_ratio,           \
            tempo_norm_v_oh,         \
            style_tag_array_oh,      \
            song_progress_oh,        \
            n_note_in_bar,           \
            drum_bar_data_reload)
    
    
# check data format
py_func_out_cqt_data,     \
py_func_out_cqt_ratio,    \
py_func_out_tempo_att,    \
py_func_out_style_att,    \
py_func_out_progress_att, \
py_func_out_n_note_att,   \
py_func_out_drum_arrange = read_pkl_function(abs_bar_idx_str_list[0])

# show data format
print ('[info] data pkg out[0] shape: {}'.format(py_func_out_cqt_data.shape))
print ('[info] data pkg out[1] shape: {}'.format(py_func_out_cqt_ratio.shape))
print ('[info] data pkg out[2] shape: {}'.format(py_func_out_tempo_att.shape))
print ('[info] data pkg out[3] shape: {}'.format(py_func_out_style_att.shape))
print ('[info] data pkg out[4] shape: {}'.format(py_func_out_progress_att.shape))
print ('[info] data pkg out[5] shape: {}'.format(py_func_out_n_note_att.shape))
print ('[info] data pkg out[6] shape: {}'.format(py_func_out_drum_arrange.shape))

### Define dataset.map function data shape
trf_out0_shape = py_func_out_cqt_data.shape
trf_out1_shape = py_func_out_cqt_ratio.shape
trf_out2_shape = py_func_out_tempo_att.shape
trf_out3_shape = py_func_out_style_att.shape
trf_out4_shape = py_func_out_progress_att.shape
trf_out5_shape = py_func_out_n_note_att.shape
trf_out6_shape = py_func_out_drum_arrange.shape

def tf_reshape_function(trf_out0, trf_out1, trf_out2, trf_out3, trf_out4, trf_out5, trf_out6):
    
    trf_out0.set_shape(trf_out0_shape)    # cqt data
    trf_out1.set_shape(trf_out1_shape)    # cqt data ratio
    trf_out2.set_shape(trf_out2_shape)    # tempo data
    trf_out3.set_shape(trf_out3_shape)    # style data
    trf_out4.set_shape(trf_out4_shape)    # song progress
    trf_out5.set_shape(trf_out5_shape)    # note number in bar
    trf_out6.set_shape(trf_out6_shape)    # drum arrange    
    
    return trf_out0, trf_out1, trf_out2, trf_out3, trf_out4, trf_out5, trf_out6

print('[info] \"dataset.map\" function is defined.')

### Define TF dataset API for test data
batch_size = 64

# extend list to match needed batch size
abs_bar_idx_str_list_ext = copy.deepcopy(abs_bar_idx_str_list)
abs_bar_idx_str_list_ext.extend(copy.deepcopy(abs_bar_idx_str_list))

extend_list_len = ((len(abs_bar_idx_str_list)//batch_size) + \
                    np.int(np.ceil((len(abs_bar_idx_str_list)%batch_size)/batch_size))) * batch_size

abs_bar_idx_str_list_ext = abs_bar_idx_str_list_ext[:extend_list_len]

orig_list_n = len(abs_bar_idx_str_list); extd_list_n = len(abs_bar_idx_str_list_ext); 
print ('[info] Original list len: {}, batch num: {}'.format(orig_list_n, orig_list_n/batch_size))
print ('[info] Extended list len: {}, batch num: {}'.format(extd_list_n, int(extd_list_n/batch_size)))

print ('[info] Total test index codes: {}'.format(len(abs_bar_idx_str_list_ext)))

darr_test_dataset = tf.data.Dataset.from_tensor_slices((abs_bar_idx_str_list_ext))
darr_test_dataset = darr_test_dataset.map(lambda index_code_test: tuple(tf.py_function(read_pkl_function,                    
                                                                                   [index_code_test],
                                                                                   [tf.float32, 
                                                                                    tf.float32, 
                                                                                    tf.float32, 
                                                                                    tf.float32, 
                                                                                    tf.float32, 
                                                                                    tf.float32, 
                                                                                    tf.float32])),
                                           num_parallel_calls=8)

darr_test_dataset = darr_test_dataset.map(tf_reshape_function, num_parallel_calls=8)
darr_test_dataset = darr_test_dataset.batch(batch_size=batch_size)

# test_iter = darr_test_dataset.make_initializable_iterator()
test_iter = tf.compat.v1.data.make_initializable_iterator(darr_test_dataset)

# get batch data
batch_bar_cqt_data_test,         \
batch_bar_cqt_ratio_test,        \
batch_bar_tempo_data_test,       \
batch_bar_style_data_test,       \
batch_bar_progress_test,         \
batch_bar_note_num_test,         \
batch_bar_arrange_test = test_iter.get_next()

# define TF-placeholder to hold note complexity adjust value
tfph_bar_add_note_num = tf.compat.v1.placeholder(tf.float32, shape=(1))

print('[info] TF test Data API is defined.')


### Define Encoder Model

# define leaky relu function
def lrelu(x, alpha=0.05):
    return tf.maximum(x, tf.multiply(x, alpha))

n_latent = 32

# define spectrogram encoder
def spec_encoder(enc_song_tempo,        # (batch_num, 10)
                 enc_style_id,          # (batch_num, 15)
                 enc_song_progress,     # (batch_num, 10)  
                 enc_spectrogram,       # (batch_num, 84, 96, 2)
                 reuse=False):
    
    with tf.compat.v1.variable_scope('spec_nn_enc', reuse=reuse):
        
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
            
        else:
            assert tf.compat.v1.get_variable_scope().reuse is False    
        
        # define song_tempo input layer
        enc_song_tempo_i_layer = tf.compat.v1.layers.dense(inputs=enc_song_tempo,
                                                 units=64,
                                                 activation=lrelu,
                                                 name='enc_nn_at1')                                                    

        # define style_id input layer
        enc_style_id_i_layer = tf.compat.v1.layers.dense(inputs=enc_style_id,
                                               units=64,
                                               activation=lrelu,
                                               name='enc_nn_at2')          
        # define song_progress input layer
        enc_song_progress_i_layer = tf.compat.v1.layers.dense(inputs=enc_song_progress,
                                                    units=64,
                                                    activation=lrelu,
                                                    name='enc_nn_at3')
        
        # make padding Batch / Height / Width / Channel 
        enc_spectrogram_pad = tf.pad(enc_spectrogram, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
                
        enc_conv_h1 = tf.nn.elu(instance_norm(conv2d(enc_spectrogram_pad, 
                                                     output_dim=48,
                                                     ks=[4,4],
                                                     s=[1,1], 
                                                     name='enc_conv1'), 'enc_bn1'))
    
        enc_conv_h2 = tf.nn.elu(instance_norm(conv2d(enc_conv_h1, 
                                                     output_dim=48,
                                                     ks=[4,4],
                                                     s=[2,2], 
                                                     name='enc_conv2'), 'enc_bn2'))
        
        enc_conv_h3 = tf.nn.elu(instance_norm(conv2d(enc_conv_h2,
                                                     output_dim=72,
                                                     ks=[4,4],
                                                     s=[2,2], 
                                                     name='enc_conv3'), 'enc_bn3'))
    
    
        # flatten conv output
        enc_convo_flat_out = tf.reshape(enc_conv_h3, [-1, np.prod(enc_conv_h3.get_shape()[1:])])
        
        # concat all decoder input layers
        enc_merged_layer = tf.concat([enc_song_tempo_i_layer,       \
                                      enc_style_id_i_layer,         \
                                      enc_song_progress_i_layer,    \
                                      enc_convo_flat_out],          \
                                     axis=1,                        \
                                     name='enc_nn_in_concat')
    
        
        enc_mlp_h1 = tf.compat.v1.layers.dense(inputs=enc_merged_layer,
                                 units=1024,
                                 activation=lrelu,
                                 name='enc_nn_mid_h1')
        
        enc_mlp_h2 = tf.compat.v1.layers.dense(inputs=enc_mlp_h1,
                                 units=1024,
                                 activation=lrelu,
                                 name='enc_nn_mid_h2')

        enc_mlp_h2m = lrelu(enc_mlp_h2 + enc_mlp_h1*0.2)
        
        enc_mlp_h3 = tf.compat.v1.layers.dense(inputs=enc_mlp_h2m,
                                     units=1024,
                                     activation=lrelu,
                                     name='enc_nn_mid_h3')
        
        enc_mlp_h3m = lrelu(enc_mlp_h3 + enc_mlp_h2m*0.2)      
        
        
        # define encoder output layer
        z_mean = tf.compat.v1.layers.dense(inputs=enc_mlp_h3m,
                                 units=n_latent,
                                 activation=None,
                                 name='enco_mean')
            
        z_std = tf.compat.v1.layers.dense(inputs=enc_mlp_h3m, 
                                units=n_latent, 
                                activation=None,
                                name='enco_std')
        
        z_epsilon = tf.random.normal(tf.stack([tf.shape(enc_mlp_h3m)[0], n_latent])) 
        
        z_latent  = z_mean + tf.multiply(z_epsilon, tf.exp(z_std * 0.5))
        
        enc_n_note_h1 = tf.compat.v1.layers.dense(inputs=enc_mlp_h3m, 
                                        units=512, 
                                        activation=lrelu,
                                        name='enc_nnp_h1')

        enc_n_note_h2 = tf.compat.v1.layers.dense(inputs=enc_n_note_h1, 
                                        units=512, 
                                        activation=lrelu,
                                        name='enc_nnp_h2')        
        
        enc_n_note_pridiction = tf.compat.v1.layers.dense(inputs=enc_n_note_h2, 
                                                units=1, 
                                                activation=tf.nn.sigmoid,
                                                name='enco_nnp_out')
        
        enc_n_note_pridiction_x256 = (enc_n_note_pridiction * 256) - 10
        
        return z_latent, z_mean, z_std, enc_n_note_pridiction_x256
    
print ('[info] Encoder define done.')

### Define Decoder Model
dec_output_size = np.prod(trf_out6_shape)   # bar_arrange, out[5] shape: (46, 16, 1)
dec_output_size_4d = [-1, trf_out6_shape[0], trf_out6_shape[1], 1]

# define leaky relu function
def lrelu(x, alpha=0.05):
    return tf.maximum(x, tf.multiply(x, alpha))

# define spectrogram encoder
def spec_decoder(dec_song_tempo,        # (batch_num, 10)
                 dec_style_id,          # (batch_num, 15)
                 dec_song_progress,     # (batch_num, 10)   
                 dec_bar_note_num,      # (batch_num, 1)
                 dec_z_sampled,         # (batch_num, 32)                 
                 reuse=False):
    
    with tf.compat.v1.variable_scope('spec_nn_dec', reuse=reuse):
        
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
            
        else:
            assert tf.compat.v1.get_variable_scope().reuse is False          
            
        # define song_tempo input layer
        dec_song_tempo_i_layer = tf.compat.v1.layers.dense(inputs=dec_song_tempo,
                                                 units=64,
                                                 activation=lrelu,
                                                 name='dec_nn_at1')                                                    

        # define style_id input layer
        dec_style_id_i_layer = tf.compat.v1.layers.dense(inputs=dec_style_id,
                                               units=64,
                                               activation=lrelu,
                                               name='dec_nn_at2')          
        # define song_progress input layer
        dec_song_progress_i_layer = tf.compat.v1.layers.dense(inputs=dec_song_progress,
                                                    units=64,
                                                    activation=lrelu,
                                                    name='dec_nn_at3')
        
        # define bar_note_num input layer
        dec_bar_note_num_limited = tf.clip_by_value(dec_bar_note_num,
                                                    0.0,
                                                    200.0)
        
        dec_bar_note_num_i_layer = tf.compat.v1.layers.dense(inputs=dec_bar_note_num_limited,
                                                   units=64,
                                                   activation=lrelu,
                                                   name='dec_nn_at4')
        
        # define z input layer
        dec_z_i_layer = tf.compat.v1.layers.dense(inputs=dec_z_sampled,
                                        units=256,
                                        activation=lrelu,
                                        name='dec_nn_at5')
        
        # concat all decoder input layers
        dec_merged_layer = tf.concat([dec_song_tempo_i_layer,       \
                                      dec_style_id_i_layer,         \
                                      dec_song_progress_i_layer,    \
                                      dec_bar_note_num_i_layer,     \
                                      dec_z_i_layer],               \
                                     axis=1,                        \
                                     name='dec_nn_in_concat')
                
        dec_mlp_h1 = tf.compat.v1.layers.dense(inputs=dec_merged_layer,
                                     units=1024,
                                     activation=lrelu,
                                     name='dec_nn_mid_h1')                                     
        
        dec_mlp_h2 = tf.compat.v1.layers.dense(inputs=dec_mlp_h1,
                                     units=1024,
                                     activation=lrelu,
                                     name='dec_nn_mid_h2')   
        
        dec_mlp_h2m = lrelu(dec_mlp_h2 + dec_mlp_h1*0.2)
        
        dec_mlp_h3 = tf.compat.v1.layers.dense(inputs=dec_mlp_h2m,
                                     units=2048,
                                     activation=lrelu,
                                     name='dec_nn_mid_h3')
        
        dec_mlp_h4 = tf.compat.v1.layers.dense(inputs=dec_mlp_h3,
                                     units=2048,
                                     activation=lrelu,
                                     name='dec_nn_mid_h4')
        
        dec_mlp_h4m = lrelu(dec_mlp_h4 + dec_mlp_h3*0.2)
        
        dec_mlp_h5 = tf.compat.v1.layers.dense(inputs=dec_mlp_h4m,
                                     units=2048,
                                     activation=lrelu,
                                     name='dec_nn_mid_h5')
        
        dec_mlp_h5m = lrelu(dec_mlp_h5 + dec_mlp_h4m*0.2)
        
        dec_mlp_h6 = tf.compat.v1.layers.dense(inputs=dec_mlp_h5m,
                                     units=2048,
                                     activation=lrelu,
                                     name='dec_nn_mid_h6')
        
        dec_mlp_h6m = lrelu(dec_mlp_h6 + dec_mlp_h5m*0.2)
        
        # final output layer use tanh
        dec_mlp_output = tf.compat.v1.layers.dense(inputs=dec_mlp_h6m,
                                         units=dec_output_size,                        
                                         activation=tf.nn.tanh,
                                         name='dec_nn_out_final')        
        
        # normalize output range to -1.5 ~ 2.5
        #dec_mlp_output_norm = (dec_mlp_output * 4.0) + 0.5        
        dec_mlp_output_norm = lrelu((dec_mlp_output * 2.0) + 0.5)
        
        # reshape data into 4d shape
        dec_output_reshape = tf.reshape(dec_mlp_output_norm, dec_output_size_4d)
        
        return dec_output_reshape

print ('[info] Decoder define done.')


### Make model connections
# connect model for testing data
processed_cqt_data_test = apply_cqt_ratio(batch_bar_cqt_data_test, batch_bar_cqt_ratio_test)
processed_cqt_data_double_layer_test = get_matx_2_layer_tf(processed_cqt_data_test)

vae_latent_z_test,     \
vae_latent_zmn_test,   \
vae_latent_zsd_test,   \
vae_note_pred_test = spec_encoder(batch_bar_tempo_data_test,                   \
                                  batch_bar_style_data_test,                   \
                                  batch_bar_progress_test,                     \
                                  processed_cqt_data_double_layer_test,        \
                                  reuse=False)

vae_drum_out_test = spec_decoder(batch_bar_tempo_data_test,                    \
                                 batch_bar_style_data_test,                    \
                                 batch_bar_progress_test,                      \
                                 vae_note_pred_test + tfph_bar_add_note_num,   \
                                 vae_latent_z_test,                            \
                                 reuse=False)

print ('[info] VAE test model is connected.')

### Define all parameters
# Define all trainable variable
t_vars = tf.compat.v1.trainable_variables()

# count model trainable variables
print('[info] Total params: {}'.format(np.sum([np.prod(v.shape) for v in t_vars])))
print('[info] Encoder params: {}'.format(np.sum([np.prod(v.shape) for v in t_vars if 'spec_nn_enc' in v.name])))
print('[info] Decoder params: {}'.format(np.sum([np.prod(v.shape) for v in t_vars if 'spec_nn_dec' in v.name])))

# collect all tf variables
nn_model_vars = [var for var in t_vars if 'spec_nn_enc' in var.name]
nn_model_vars.extend([var for var in t_vars if 'spec_nn_dec' in var.name])

print('\n[info] trainable variable: ')
print([var.name for var in nn_model_vars])

### Run testing loop
show_info_epoch = 1; show_info_batch = 1;

print ("[info] Testing cell running...")
print ('[info] ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')

# init tensorflow variables
init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver(var_list=nn_model_vars)
darr_model_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
darr_model_config.gpu_options.allow_growth = True

# run TF session here
with tf.compat.v1.Session(config=darr_model_config) as sess:
    
    start_time = datetime.datetime.now()
    
    sess.run(init)
    
    # reload model
    saver.restore(sess, './drum_generator_model/darr_model.ckpt')
    print ('[info] Model parameters loaded.')
    
    epoch_target = 1
    
    # run Epoch loop here
    for add_note_v in bar_add_note_num_list:

        print ('\n')
        print ('[info] Add note complexity: {}'.format(add_note_v))
        print ('[info] Test start...\n')
        
        sess.run(test_iter.initializer)

        note_score_test_list = []

        session_bar_cqt_data_test_list = []
        session_bar_note_num_test_list = []
        session_bar_note_num_pred_test_list = []
        session_bar_arrange_test_list = []
        session_darr_output_test_list = []
        
        batch_target_test = np.int(len(abs_bar_idx_str_list_ext)/batch_size)        
        batch_runned_test = 0 
        
        # run batch loop here
        for batch_idx in range(0, batch_target_test):            
            
            # run model testing            
            session_bar_cqt_data_test,             \
            session_bar_style_data_test,           \
            session_bar_tempo_data_test,           \
            session_bar_note_num_test,             \
            session_bar_progress_test,             \
            session_bar_arrange_test,              \
            session_bar_note_num_pred_test,        \
            session_darr_output_test = sess.run([batch_bar_cqt_data_test,  \
                                       batch_bar_style_data_test,          \
                                       batch_bar_tempo_data_test,          \
                                       batch_bar_note_num_test,            \
                                       batch_bar_progress_test,            \
                                       batch_bar_arrange_test,             \
                                       vae_note_pred_test,                 \
                                       vae_drum_out_test],                 \
                                      feed_dict={tfph_bar_add_note_num: np.array([add_note_v])})
            
            # get correct shape of data
            session_bar_cqt_data_test = session_bar_cqt_data_test.copy()[:,:,:,0]
            session_bar_arrange_test = session_bar_arrange_test.copy()[:,:,:,0]
            session_darr_output_test = session_darr_output_test.copy()[:,:,:,0]
            
            # calculate note score
            session_darr_output_test_bin = np.where(session_darr_output_test>=0.5,
                                                    np.ones_like(session_darr_output_test),
                                                    np.zeros_like(session_darr_output_test))  

            note_score_test = 1.0 - np.sum(np.abs(                        \
                session_bar_arrange_test - session_darr_output_test_bin))/np.prod(session_bar_arrange_test.shape)
            
            note_score_test_list.append(note_score_test)
            
            # record every batch data
            session_bar_cqt_data_test_list.append(session_bar_cqt_data_test)
            session_bar_note_num_test_list.append(session_bar_note_num_test)
            session_bar_note_num_pred_test_list.append(session_bar_note_num_pred_test)
            session_bar_arrange_test_list.append(session_bar_arrange_test)
            session_darr_output_test_list.append(session_darr_output_test)
                
            # record runned batch
            batch_runned_test += 1
            
            if (batch_runned_test%show_info_batch)==0:
                out_msg = "[info] Batch done: [ {:3d} / {:3d} ]".format(batch_runned_test, batch_target_test)
                out_msg += ",  Note score: {:.2f} %".format(100*np.mean(note_score_test_list[-show_info_batch:]))
                print (out_msg)             
            
        delta_time = datetime.datetime.now() - start_time
        
        out_msg  = "\n[info] Test note score(Avg.): {:.2f} %".format(100*np.mean(note_score_test_list))
        out_msg += "\n[info] test error notes per bar({} notes): {:.2f}".format(dec_output_size, 
                                                                                (1-np.mean(note_score_test_list))*dec_output_size)
        out_msg += "\n[info] Elapse Time: {}".format(str(delta_time)[:-7])        
        print (out_msg)
        
        # save calculation result
        cqt_data_ary   = np.concatenate(session_bar_cqt_data_test_list, axis=0)[:len(abs_bar_idx_str_list),:]
        drum_original  = np.concatenate(session_bar_arrange_test_list, axis=0)[:len(abs_bar_idx_str_list),:]
        drum_predicted = np.concatenate(session_darr_output_test_list, axis=0)[:len(abs_bar_idx_str_list),:]
        dump_data_pkg = [cqt_data_ary, drum_original, drum_predicted]

        print("[info] CQT data shape: {}".format(cqt_data_ary.shape))
        print("[info] Drum data shape (Original): {}".format(drum_original.shape))
        print("[info] Drum data shape (predicted): {}".format(drum_predicted.shape))

        dump_file_name = './model_out_result_add_note_{:0>2}.pkl'.format(add_note_v)
        with open(dump_file_name, 'wb') as pkl_file:
            pickle.dump(dump_data_pkg, pkl_file)

        print ('[info] Saved file:  \"{}\"'.format(dump_file_name))
        
        # test data session end            
        print ('[info] Test session is finished.')   
    
# show process is end
print ("\n\n[info] All testing process is finished.")
print ("[info] " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


### save s100 p00n/p03n/p06n/p12n/p20n test result

test_result_train_pp = [session_bar_cqt_data_train_list,
                        session_bar_note_num_train_list,
                        session_bar_note_num_pred_train_list,
                        session_bar_arrange_train_list,
                        session_darr_output_train_list]


test_s100_result_fname = './model_test_result_bm24_vaegan/{}/rc_loss_{}/'.format(chkpt_ver, rc_loss_ver)
test_s100_result_fname += 'bm24_{}_result_pkg.pkl'.format(add_note_ver)

ensure_dir(test_s100_result_fname)
with open(test_s100_result_fname, 'wb') as pkl_file:
    pickle.dump(test_result_train_pp, pkl_file)
    
print ('[info] s100 {}, {}, {} test result saved.'.format(chkpt_ver, rc_loss_ver, add_note_ver))
print ('[info] saved file:  \"{}\"'.format(test_s100_result_fname))
