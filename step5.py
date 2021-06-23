import librosa, IPython, datetime, time, os, sys, copy, glob, pickle
import numpy as np
from time import gmtime, strftime
from IPython.display import Image
import pypianoroll
import matplotlib.pyplot as plt
# %matplotlib inline

# show version info
print ("[info] Current Time:     " + datetime.datetime.now().strftime('%Y/%m/%d  %H:%M:%S'))
print ("[info] Python Version:   " + sys.version.split('\n')[0].split(' ')[0])
print ("[info] Working Dir:      " + os.getcwd()+'/')

def ensure_dir(file_path):
    ed_directory = os.path.dirname(file_path)
    if not os.path.exists(ed_directory):
        os.makedirs(ed_directory)


### Read all song/bar index code
with open('./pre_processed_data/abs_bar_idx_str_list.pkl', 'rb') as pkl_file:      
    abs_bar_idx_str_list = pickle.load(pkl_file)
    
print ('[info] List of [song/bar] data is loaded.')
print ('[info] Total bars: {}'.format(len(abs_bar_idx_str_list)))
print ('[info] First 5 bar code: {}'.format(abs_bar_idx_str_list[:5]))
print ('[info] Last  5 bar code: {}'.format(abs_bar_idx_str_list[-5:]))


# Define function to get complete single song index (start, end)
song_index_in_list = np.unique([x.split('_')[0] for x in abs_bar_idx_str_list]).tolist()

def get_test_song_abs_idx(pick_song_index):

    song_index_all_bars = [x for x in abs_bar_idx_str_list if x[0:5]==song_index_in_list[pick_song_index]]
    bar_idx_start = abs_bar_idx_str_list.index(song_index_all_bars[0])
    bar_idx_end = abs_bar_idx_str_list.index(song_index_all_bars[-1])
    
    return ([bar_idx_start, bar_idx_end+1])


for get_song_idx in range(0, 24):
    print('[info] Song idx: {:2d},   Start:{:4d},   End: {}'.format(get_song_idx,
                                                                    get_test_song_abs_idx(get_song_idx)[0],
                                                                    get_test_song_abs_idx(get_song_idx)[1]))


### Read all test result
model_result_flist = np.sort(glob.glob('./model_out_result_add_note_*.pkl', recursive=True)).tolist()

model_result_binary_list = []
add_note_ver_list = []

for model_result_file in model_result_flist:

    with open(model_result_file, 'rb') as pkl_file:
        model_result_pkg = pickle.load(pkl_file)
        
    model_result_binary = np.where(model_result_pkg[2] > 0.5,
                                   np.ones_like(model_result_pkg[2]),
                                   np.zeros_like(model_result_pkg[2]))
    
    print ('[info] \'{}\' is reloaded.'.format(model_result_file))
    print ('[info] Data shape: {}'.format(model_result_binary.shape))
        
    model_result_binary_list.append(model_result_binary)
    
    add_note_ver = model_result_file.split('.')[-2][-2:]
    add_note_ver_list.append(add_note_ver)

print ('\n[info] {} files are reloaded.'.format(len(model_result_flist)))


### Reload all original MIDI object
class midi_track(object):
    def __init__(self):
        self.file_name = ""
        self.pmidi_data = []
        self.pmidi_all_tracks_data = []
        self.pmidi_no_drum_data = []
        self.pmidi_drum_only_data = []
        self.tempo = 0        
        self.downbeats_list_fixed = []
        self.bar_range_list_fixed = []
        self.drum_bar_list = []
        self.drum_bar_list_bin = []
        self.drum_bar_note_num = []
#print ('MIDI track object is defined.')

obj_file_name = './pre_processed_data/proc_midi_object.pkl'
with open(obj_file_name, 'rb') as pkl_file:
    midi_obj_list = pickle.load(pkl_file)
    
print('[info] All MIDI objects: {}'.format(len(midi_obj_list)))


### Define original MIDI drum rebuild function (96, 128)
# keep 99 % of all instrument count (total 46 insts)
selected_inst_list_46 = [27, 28, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, \
                         51, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70, 73, \
                         74, 75, 76, 77, 80, 81, 82, 83, 85, 87]
print ('[info] # of keeped Insts: {}'.format(len(selected_inst_list_46)))

def get_odrum_shape(drum_ary_in):    
    odrum_data = np.zeros([96, 128])
    for x in range(0, drum_ary_in.shape[0]):
        for y in range(0, drum_ary_in.shape[1]):            
            pix_value = drum_ary_in[x,y]
            if pix_value>0.5:
                odrum_data[y*6, selected_inst_list_46[x]] = 100
            
    return (odrum_data)

print ('[info] get_odrum_shape is defined.')


### Load original midi data
all_tracks_mid_flist = np.sort(glob.glob('./input_midi/**/*.mid', recursive=True)).tolist()
all_tracks_mid_flist = np.sort([x.replace(' ','') for x in all_tracks_mid_flist if "all_tracks.mid" in x]).tolist()
print ('[info] Total files: {}'.format(len(all_tracks_mid_flist)))
for x in all_tracks_mid_flist[:]: print ('  ' + x)

### Loop all songs and save corresponding MIDI files
for x_idx, pick_song_index in enumerate(song_index_in_list):

    print ('[info] Start processing song: {} ...'.format(x_idx+1))

    # get complete single song index data
    abs_idx_start, abs_idx_end = get_test_song_abs_idx(x_idx)
    
    abs_song_idx = pick_song_index

    print ('[info] Song index: {}'.format(abs_song_idx))
    print ('[info] Song bars: {}'.format(abs_idx_end - abs_idx_start))
    print ('[info] start\end bar index:  {}\{}'.format(abs_idx_start, abs_idx_end))
    #print ('[info] Abs end index: {}'.format(abs_idx_end))
    #print('')

    # plot complete single song drum arrangement
    bar_idx_start = abs_idx_start
    bar_idx_end = abs_idx_end

    model_darr_odrm_ary_list = []
    
    for pch_ver in range(0, len(model_result_binary_list)):
    
        model_darr_list = []

        for bar_idx in range(bar_idx_start, bar_idx_end):
        
            plot_model_out_darr = model_result_binary_list[pch_ver][bar_idx,:,:]
        
            model_darr_list.append(plot_model_out_darr)
        

        # convert drum data into original shape (96, 128)
        model_darr_odrm_list = [get_odrum_shape(x) for x in model_darr_list]
        model_darr_odrm_ary = np.concatenate(model_darr_odrm_list, axis=0)
        #print(model_darr_odrm_ary.shape)

        model_darr_odrm_ary_list.append(model_darr_odrm_ary)

    
    #Get original NPZ file name
    original_midi_file_path = all_tracks_mid_flist[x_idx]
    pypiano_obj = pypianoroll.parse(original_midi_file_path, beat_resolution=24, name='original_track')
    #ptymidi_obj = pypiano_obj.to_pretty_midi()
    mtrack_data = pypiano_obj
    
    for pch_idx in range(0, len(model_result_binary_list)):
        
        # write drum notes in multitrack object
        mtrack_data.append_track(track=None, 
                                 pianoroll=model_darr_odrm_ary_list[pch_idx], 
                                 program=pch_idx+1, 
                                 is_drum=True,
                                 name='Drums_{}'.format(add_note_ver_list[pch_idx]))

    # transfer data into pretty midi format
    pmidi_data = mtrack_data.to_pretty_midi(constant_tempo=None)

    # print instruments
    print ('[info] Show {} Insts...'.format(len(pmidi_data.instruments)))
    for x in pmidi_data.instruments:
        print ('[info] MIDI ' + str(x))
    print('')

    # make all notes in Drums2 velocity=99
    for instrument in pmidi_data.instruments:
        #if instrument.program==5:
        if instrument.is_drum:
            for note in instrument.notes:
                note.velocity = 120
        else:
            for note in instrument.notes:
                note.velocity = 50            


    song_name_tmp = all_tracks_mid_flist[x_idx].split('/')[-1][:-15] + '_merged'
                
    # set midi file name to write
    midi_file_name = './output_midi/{}.mid'.format(song_name_tmp)

    # create folder if not exist
    ensure_dir(midi_file_name)

    # write midi file
    pmidi_data.write(midi_file_name)
    print ('[info] \"{}\" is saved.\n\n'.format(midi_file_name))
    
print ('[info] All {} files are saved.'.format(len(song_index_in_list)))
