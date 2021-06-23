import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import librosa, IPython, datetime, time, os, sys, copy, dill, pickle, mir_eval, glob
import numpy as np
import pandas as pd
from time import gmtime, strftime
import tensorflow as tf
from matplotlib import pyplot as plt
from tf_ops import *

import matplotlib.pyplot as plt
# %matplotlib inline

print ("[info] Current Time:     " + datetime.datetime.now().strftime('%Y/%m/%d  %H:%M:%S'))
print ("[info] Python Version:   " + sys.version.split('\n')[0].split(' ')[0])
print ("[info] Working Dir:      " + os.getcwd()+'/')

# enable gpu usage constraint here
fixed_gpu_usage = 1
selected_gpu_id = 0

# if gpu usage is constraint, limit certain gpu for use
if (fixed_gpu_usage == 1):
    # set available GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        # list GPU sequence by PCI bus GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(selected_gpu_id)  

    # check available GPU
    from tensorflow.python.client import device_lib
    for x in range(1, len(device_lib.list_local_devices())):
        print ("[info] GPU " + device_lib.list_local_devices()[x].physical_device_desc)

test_file_names_list = np.sort(glob.glob('./pre_processed_data/merged_ssm_pkg/*.pkl', recursive=True)).tolist()
print ('[info] Total test files: {}'.format(len(test_file_names_list)))

total_test_files_num = len(test_file_names_list)   

def get_batch_ssm_data(get_batch_num=total_test_files_num):

    batch_file_names_list = copy.deepcopy([test_file_names_list[x] for x in list(range(0, total_test_files_num))])
    
    cqt_ssm_batch_list = []
    drum_ssm_batch_list = []
    song_bars_len_list = []    
    
    for file_name_read in batch_file_names_list:
        
        with open(file_name_read, 'rb') as pkl_file:        
            reload_data_package = pickle.load(pkl_file)
            
        cqt_ssm_data = reload_data_package[0].astype(np.float32)
        drum_ssm_data = reload_data_package[1].astype(np.float32)          
        song_bars_len = reload_data_package[4]
        
        # save data into list
        cqt_ssm_batch_list.append(cqt_ssm_data)
        drum_ssm_batch_list.append(drum_ssm_data)
        song_bars_len_list.append(song_bars_len)
    
    # merge batch data
    ssm_batch_output = [cqt_ssm_batch_list, 
                        drum_ssm_batch_list, 
                        song_bars_len_list]
    
    return(ssm_batch_output)

print ('[info] Data loader define done.')


print('')

picked_song_idx = 20
print ('[info] CQT SSM shape: {}'.format(get_batch_ssm_data()[0][picked_song_idx].shape))
print ('[info] Drum SSM shape: {}'.format(get_batch_ssm_data()[1][picked_song_idx].shape))
print ('[info] Song bars len shape: {}'.format(get_batch_ssm_data()[2][picked_song_idx]))

# get CQT SSM shape
cqt_ssm_shape = get_batch_ssm_data()[0][0].shape

tf.compat.v1.disable_eager_execution()
cqt_ssm_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, 
                                               cqt_ssm_shape[0], 
                                               cqt_ssm_shape[1], 
                                               cqt_ssm_shape[2]])

# define drum_arranger
def ssm_arranger(sarr_cqt_ssm_input,                 # input data shape = (batch_num, 256, 256, 3)
                 reuse=False):
                
    with tf.compat.v1.variable_scope('sarr', reuse=reuse):
        
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        else:
            assert tf.compat.v1.get_variable_scope().reuse is False
        
        # set basic parameters
        fmap_layer = 48
        #keep_prob = 0.8
        unet_output_c_dim = 1
        
        e0 = sarr_cqt_ssm_input
            
        # image is (256 x 256 x input_c_dim)
        e1 = tf.nn.elu(instance_norm(conv2d(e0, fmap_layer*1, name='g_e1_conv'), 'g_bn_e1'))
        # e1 is (128 x 128 x self.gf_dim)
        e2 = tf.nn.elu(instance_norm(conv2d(e1, fmap_layer*2, name='g_e2_conv'), 'g_bn_e2'))
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = tf.nn.elu(instance_norm(conv2d(e2, fmap_layer*4, name='g_e3_conv'), 'g_bn_e3'))
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = tf.nn.elu(instance_norm(conv2d(e3, fmap_layer*4, name='g_e4_conv'), 'g_bn_e4'))
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = tf.nn.elu(instance_norm(conv2d(e4, fmap_layer*4, name='g_e5_conv'), 'g_bn_e5'))
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = tf.nn.elu(instance_norm(conv2d(e5, fmap_layer*4, name='g_e6_conv'), 'g_bn_e6'))
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = tf.nn.elu(instance_norm(conv2d(e6, fmap_layer*4, name='g_e7_conv'), 'g_bn_e7'))
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = tf.nn.elu(instance_norm(conv2d(e7, fmap_layer*4, name='g_e8_conv'), 'g_bn_e8'))
        # e8 is (1 x 1 x self.gf_dim*8)

        d1 = tf.nn.elu(instance_norm(deconv2d(e8, fmap_layer*4, name='g_d1'), 'g_bn_d1'))
        #d1 = tf.nn.dropout(d1, keep_prob)
        d1 = tf.concat([d1, e7*0.2], axis=3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = tf.nn.elu(instance_norm(deconv2d(d1, fmap_layer*4, name='g_d2'), 'g_bn_d2'))
        #d2 = tf.nn.dropout(d2, keep_prob)
        d2 = tf.concat([d2, e6*0.20], axis=3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = tf.nn.elu(instance_norm(deconv2d(d2, fmap_layer*4, name='g_d3'), 'g_bn_d3'))
        #d3 = tf.nn.dropout(d3, keep_prob)
        d3 = tf.concat([d3, e5*0.20], axis=3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = tf.nn.elu(instance_norm(deconv2d(d3, fmap_layer*4, name='g_d4'), 'g_bn_d4'))
        d4 = tf.concat([d4, e4*0.20], axis=3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = tf.nn.elu(instance_norm(deconv2d(d4, fmap_layer*4, name='g_d5'), 'g_bn_d5'))
        d5 = tf.concat([d5, e3*0.10], axis=3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = tf.nn.elu(instance_norm(deconv2d(d5, fmap_layer*2, name='g_d6'), 'g_bn_d6'))
        d6 = tf.concat([d6, e2*0.05], axis=3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = tf.nn.elu(instance_norm(deconv2d(d6, fmap_layer*2, name='g_d7'), 'g_bn_d7'))
        d7 = tf.concat([d7, e1*0.025], axis=3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        #d8 = deconv2d(tf.nn.elu(d7), unet_output_c_dim, name='g_d8')
        d8 = deconv2d(d7, unet_output_c_dim, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)
        
        # finally get output
        #sarr_output = tf.nn.tanh(d8) * 200
        sarr_output = d8
        
        return (sarr_output)
    
print ('Model define done.')

# connect model data-flow
sarr_output_data = ssm_arranger(cqt_ssm_ph, reuse=False)

print('[info] model data-flow wire are connected.')

# Define all trainable variable
t_vars = tf.compat.v1.trainable_variables()
sarr_vars = [var for var in t_vars if 'sarr' in var.name]

print('[info] model vars are defined.')

# count model trainable variables
# print('[info] Total params: {}'.format(np.sum([np.prod(v.shape) for v in t_vars]).value))
print('[info] Total params: {}'.format(np.sum([np.prod(v.shape) for v in t_vars])))

print('[info] trainable variables: \n')
print([var.name for var in sarr_vars])

# test ssm model arrangement
chkpt_ver = 'v13'

saver = tf.compat.v1.train.Saver(var_list=sarr_vars)

sarr_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
sarr_config.gpu_options.allow_growth = True

with tf.compat.v1.Session(config=sarr_config) as sess:
    
    saver.restore(sess, './ssm_generator_model/sarr_model.ckpt'.format(chkpt_ver, chkpt_ver))
    print ('[info] Model parameters are loaded.\n')    

    test_cqt_data_list = []
    test_drum_data_sample_gt_list = []
    test_drum_data_sample_list = []
    test_song_length_data_list = []
    
    run_loops_num = 1
    
    print ('[info] Start testing...')
    print (datetime.datetime.now().strftime('[info] %Y-%m-%d %H:%M:%S') + '\n')
    
    for calculate_score_idx in range(0, run_loops_num):        

        # get test data for test
        test_batch_ssm_data_tmp = get_batch_ssm_data()                
        # get train data for test                
        test_cqt_ssm_data = np.array(test_batch_ssm_data_tmp[0])     # (batch_num_test, 256, 256, 3)
        
        # get batch ground-truth data
        test_drum_ssm_data = np.array(test_batch_ssm_data_tmp[1])    # (batch_num_test, 256, 256, 3)
        
        # get song length data     
        test_song_len_data = np.array(test_batch_ssm_data_tmp[2])    # (batch_num, 1)             
        
        # get drum arrangement test result
        test_sarr_sample = sess.run(ssm_arranger(cqt_ssm_ph, reuse=True),
                                                   feed_dict={cqt_ssm_ph:     test_cqt_ssm_data, 
                                                              #drum_ssm_ph:    test_drum_ssm_data,
                                                              #ssm_mask_ph:    test_ssm_mask_data,
                                                              #pix_mean_ph:    test_pix_mean_data
                                                              })

        test_cqt_data_list.append(test_cqt_ssm_data)
        test_drum_data_sample_gt_list.append(test_drum_ssm_data)
        test_drum_data_sample_list.append(test_sarr_sample)
        test_song_length_data_list.append(test_song_len_data)    
        
                                                                                                                         
        # show progress
        #if (calculate_score_idx+1)%5==0:
        write_msg = '[info] Test loop: [ {} / {} ]\n'.format(calculate_score_idx+1, run_loops_num)
        #    write_msg += 'Training Loss: {:.6f} ,  '.format(np.mean(tr_loss_value))
        #    write_msg += 'Testing Loss: {:.6f}'.format(np.mean(te_loss_value))
        print (write_msg)
            

# total Avg. show score
print ("[info] Drum SSM arrange test is finished.")
print ("[info] Songs tested: {}".format(len(test_drum_data_sample_list[0])))
print (datetime.datetime.now().strftime('[info] %Y-%m-%d %H:%M:%S') + '\n')

###################################################
# Define function to extract rectangular ssm data #
###################################################

# rotate SSM, crop to original size
def get_extracted_full_ssm(ges_ssm_input, ges_song_len_input):
    
    ges_output = ges_ssm_input[0:ges_song_len_input, 0:ges_song_len_input]
    ges_output_final = copy.deepcopy(ges_output)
    
    return (ges_output_final)

# rotate SSM, crop to original size
def get_extracted_triangle_ssm(ges_ssm_input, ges_song_len_input):
    
    ges_output = ges_ssm_input[0:ges_song_len_input, 0:ges_song_len_input]
    ges_output_triu_1 = np.triu(ges_output, k=0)
    ges_output_triu_2 = np.rot90(np.flipud(np.triu(ges_output, k=1)), k=-1)
    ges_output_final = copy.deepcopy(ges_output_triu_1 + ges_output_triu_2)
    
    return (ges_output_final)

import cv2, IPython, os

def resize_ssm_762(rf7_input_figure):
    
    save_data = rf7_input_figure
    save_file_name = './saving_tmp_file.png'

    fig = plt.figure(figsize=[8,8])
    ax = fig.add_subplot(111)

    ax.imshow(save_data,
              origin='lower', 
              cmap='hot')

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    plt.savefig(save_file_name,
                dpi=50,
                bbox_inches='tight',
                pad_inches=0)

    #plt.show()
    plt.close()

    #IPython.display.clear_output()

    img_readback = cv2.imread(save_file_name)
    
    os.remove(save_file_name)
    
    img_readback = np.mean(img_readback, axis=-1)
    
    return(-img_readback)

print ('function define done.')

show_songs = total_test_files_num
#show_songs = 2

batch_idx = 0

batch_idx_list = list(range(total_test_files_num))

#np.random.shuffle(batch_idx_list)

song_idx_list = []
song_bars_num_list = []

# loop over songs to collect SSM data
for k, song_idx in enumerate(batch_idx_list[:show_songs]):

    # get song data
    cqt_trangle_ssm =    copy.copy(test_cqt_data_list[batch_idx][song_idx][:,:,0])
    drum_trangle_ssm =   copy.copy(test_drum_data_sample_gt_list[batch_idx][song_idx][:,:,0])
    drum_model_ssm =     copy.copy(test_drum_data_sample_list[batch_idx][song_idx][:,:,0])
    song_bars_num =      copy.copy(test_song_length_data_list[batch_idx][song_idx])

    cqt_restored_ssm =         resize_ssm_762(get_extracted_full_ssm(cqt_trangle_ssm, song_bars_num))
    drum_restored_ssm =        resize_ssm_762(get_extracted_full_ssm(drum_trangle_ssm, song_bars_num))
    drum_model_restored_ssm =  resize_ssm_762(get_extracted_triangle_ssm(drum_model_ssm, song_bars_num))

    cqt_drum_model_ssm = np.hstack([cqt_restored_ssm, 
                                    drum_restored_ssm, 
                                    drum_model_restored_ssm])
    
    if k==0:        
        cqt_drum_model_ssm_all = cqt_drum_model_ssm
        
    else:        
        cqt_drum_model_ssm_all = np.vstack([cqt_drum_model_ssm_all, 
                                            cqt_drum_model_ssm])


    song_idx_list.append(song_idx)
    song_bars_num_list.append(song_bars_num)

    
# show songs info
for k, song_idx in enumerate(test_song_length_data_list[0].tolist()[:show_songs]):
    print ('Song idx: {}, length: {}'.format(k, song_idx))

print ('\n              ----melodic SSM----                  ----original drm SSM----               ----generated drum SSM----')
# # plot actual song SSM
# plt.figure(figsize=(8*2, 8*show_songs))
# plt.imshow(cqt_drum_model_ssm_all, cmap='hot')
# plt.show()


# Save generated SSM

n_bars_list_for_save = []
cqt_ssm_list_for_save = []
drum_ssm_list_for_save = []
drum_model_ssm_list_for_save = []


batch_idx = 0

for song_idx in range(0, total_test_files_num):
    
    song_bars_num =      copy.copy(test_song_length_data_list[batch_idx][song_idx])
    
    cqt_trangle_ssm =    copy.copy(test_cqt_data_list[batch_idx][song_idx][:,:,0])
    drum_trangle_ssm =   copy.copy(test_drum_data_sample_gt_list[batch_idx][song_idx][:,:,0])
    drum_model_ssm =     copy.copy(test_drum_data_sample_list[batch_idx][song_idx][:,:,0])    
    
    extracted_cqt_ssm =          get_extracted_triangle_ssm(cqt_trangle_ssm, song_bars_num)
    extracted_drum_ssm =         get_extracted_full_ssm(drum_trangle_ssm, song_bars_num)
    extracted_drum_model_ssm =   get_extracted_triangle_ssm(drum_model_ssm, song_bars_num)
    
    print ('Song: {}, Bars: {}, SSM shape: {}'.format(song_idx+1, song_bars_num, extracted_drum_model_ssm.shape))
    
    n_bars_list_for_save.append(song_bars_num)
    cqt_ssm_list_for_save.append(extracted_cqt_ssm)
    drum_ssm_list_for_save.append(extracted_drum_ssm)
    drum_model_ssm_list_for_save.append(extracted_drum_model_ssm)
    
print ('\n[info] conversion done')

merged_ssm_data = [n_bars_list_for_save,
                   cqt_ssm_list_for_save,                   
                   drum_ssm_list_for_save,
                   drum_model_ssm_list_for_save]

save_ssm_pkl_file_name = './pre_processed_data/model_out_drum_ssm_pkg.pkl'

with open(save_ssm_pkl_file_name, 'wb') as pkl_file:
    pickle.dump(merged_ssm_data, pkl_file)
    
print ('\n[info] All SSMs are saved.')