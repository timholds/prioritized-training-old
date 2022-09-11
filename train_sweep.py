import opendatasets as od
import numpy as np
import tensorflow as tf
import statistics
import json
import datetime
import subprocess
import psutil
import threading
import time

from queue import Queue
from multiprocessing.pool import ThreadPool


from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from keras.losses import mse, categorical_crossentropy

import random
from numba import cuda

from target_model import ConvModel

tf.config.run_functions_eagerly(True)
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# tf.data.experimental.enable_debug_mode()
# for gpu in tf.config.list_physical_devices('GPU'):
#   # tf.config.experimental.set_memory_growth(gpu, True)
#   tf.config.set_soft_device_placement(True)

# ------------ Utility Functions ------------ 

def download_qmnist():
    ''' 120k QMNIST downloads to /qmnist-the-extended-mnist-dataset-120k-images/'''
    od.download('https://www.kaggle.com/datasets/fedesoriano/qmnist-the-extended-mnist-dataset-120k-images?resource=download')
    
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
  
def prep_data(images, labels, num_classes=10):#, test_per, holdout_per):
    print(images.shape)
    print(labels.shape)

    x_train   = images[:50000]
    x_test    = images[50000: 70000]
    x_holdout = images[70000:]

    y_train   = labels[:50000]
    y_test    = labels[50000: 70000]
    y_holdout = labels[70000:]

    x_train   = x_train  .astype('float32') / 255
    x_test    = x_test   .astype('float32') / 255
    x_holdout = x_holdout.astype('float32') / 255

    # Make sure images have shape (28, 28, 1)
    x_train    = np.expand_dims(x_train,   -1)
    x_test     = np.expand_dims(x_test,    -1)
    x_holdout  = np.expand_dims(x_holdout, -1)

    print('encoding as one hot')
    y_train   = keras.utils.to_categorical(y_train, num_classes)
    y_test    = keras.utils.to_categorical(y_test,  num_classes)
    y_holdout = keras.utils.to_categorical(y_holdout,  num_classes)

    return (x_train, y_train), (x_test, y_test), (x_holdout, y_holdout)

def compile_model(model, loss='categorical_crossentropy'):
    metrics = ['accuracy', 'mse', 'categorical_crossentropy']
    opt = 'adam'
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    return model

def fit(model, x_train, y_train, batch_size, epochs, verbose=0, shuffle=True):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=.1, verbose=verbose, shuffle=shuffle)

# Step 2: create the il loss dict by evaluating the holdout model on the training data
def make_il_loss_dict(model, x_train, y_train, loss='categorical_crossentropy'):
    ''' Make sure this is done before shuffling the data'''
    from keras.losses import mse, categorical_crossentropy
  
    preds = model.predict(x_train, batch_size=2048, use_multiprocessing=True)
    if loss == 'categorical_crossentropy':
        loss_vals = mse(preds, y_train).numpy()
    elif loss == 'mse':
        loss_vals = categorical_crossentropy(preds, y_train).numpy()

    il_loss_dict = {}
    for i, loss_val in enumerate(loss_vals):
        il_loss_dict[i] = loss_val

    return il_loss_dict

def shuffle_train_set_with_idx(x, y):
    image_label_dict = {}
    assert len(x) == len(y), 'length of x and y should be the same'
    for k in range(len(x)):
        image_label_dict[k] = {'x': x[k], 'y': y[k]}

    keys = list(image_label_dict.keys())
    random.shuffle(keys)
    image_label_dict_shuffled = {key: image_label_dict[key] for key in keys}
    
    return image_label_dict_shuffled

def load_batch_into_queue(model, image_label_dict_shuffled, q, i, batch_size, null_hypothesis):
    print('starting putting batch into queue {} to {}'.format(i, i+batch_size))
    t0 = time.time()
    idxs               = [i[0]      for i in list(image_label_dict_shuffled.items())[i : i + big_batch_size]]
    x_batch_candidates = [i[1]['x'] for i in list(image_label_dict_shuffled.items())[i : i + big_batch_size]]
    y_batch_candidates = [i[1]['y'] for i in list(image_label_dict_shuffled.items())[i : i + big_batch_size]]

    if null_hypothesis:
        x_batch = np.array([image_label_dict_shuffled[x]['x'] for x in range(i, i+small_batch_size)])
        y_batch = np.array([image_label_dict_shuffled[x]['y'] for x in range(i, i+small_batch_size)])
    else:
        # # Evaluate the current model against the candidate batch and make a dict of frame idx: loss value
        #cur_candidate_loss_dict = eval_cur_model_on_batch(conv_model, x_batch_candidates, y_batch_candidates, idxs)
        preds = model.predict(np.array(x_batch_candidates), batch_size=small_batch_size, use_multiprocessing=True)
        loss_vals = categorical_crossentropy(preds, y_batch_candidates).numpy()
        cur_model_batch_loss_dict = {}
        for idx, loss_val in list(zip(idxs, loss_vals)):
            cur_model_batch_loss_dict[idx] = loss_val
    
        # Calculate RHO loss using the frame indices in both training loss dict and IL loss dict
        rho_loss = {}
        for frame_idx in cur_model_batch_loss_dict.keys():
            rho_loss[frame_idx] = cur_model_batch_loss_dict[frame_idx] - il_model_loss_dict[frame_idx]

        # Select the training batch using using the frames with highest RHO loss
        rho_loss_sorted = sorted(rho_loss.items(), key=lambda x: x[1])
        frame_idxs = [x[0] for x in rho_loss_sorted]

        # Select the training batch using the indices of the ideal frames from the subbatch
        x_batch = np.array([image_label_dict_shuffled[x]['x'] for x in frame_idxs])
        y_batch = np.array([image_label_dict_shuffled[x]['y'] for x in frame_idxs])


    q.put([x_batch, y_batch])
    t1 = time.time()
    print('took {}s to put batch into queue {} to {}'.format(t1-t0, i, i+batch_size))
    return q

def log_util_usage():
    print('\n *** logging utility usage ***\n')
    mem_data = []
    cpu_data = []
    gpu_mem_data = []
    while True:
        t0 = time.time()
        cpu_data.append(psutil.cpu_percent())
        json_content = {"datasets": {"log_cpu_util": cpu_data},
                        "type": "line-chart",
                        "version": 1,
                        "xAxisLabel": "Minute",
                        "yAxisLabel": "Percent"}

        with open('cpu_usage.json', "w+") as f:
            json.dump(json_content, f)

        mem_data.append(psutil.virtual_memory().used / (2 ** 20))
        json_content = {"datasets": {"log_memory": mem_data},
                        "type": "line-chart",
                        "version": 1,
                        "xAxisLabel": "Minute",
                        "yAxisLabel": "Memory Used (MiB)"}

        with open('cpu_memory.json', "w+") as f:
            json.dump(json_content, f)

        try:
            nvidia_smi_proc = subprocess.Popen("nvidia-smi --query-gpu=memory.used --format=csv", stdout=subprocess.PIPE, shell=True)
            stdout, _= nvidia_smi_proc.communicate(timeout=2)
            gpu_mem_used = [line for line in stdout.decode().split("\n") if len(line) > 2][1:]
            gpu_mem_used = [int(line.split(" ")[0]) for line in gpu_mem_used]
            total_gpu_mem_used = sum(gpu_mem_used)
            gpu_mem_data.append(total_gpu_mem_used)
        except (UnicodeDecodeError, IndexError, ValueError):
            gpu_mem_data.append(gpu_mem_data[-1])

        json_content = {"datasets": {"log_gpu_memory": gpu_mem_data},
                        "type": "line-chart",
                        "version": 1,
                        "xAxisLabel": "Minute",
                        "yAxisLabel": "GPU Memory Used (MiB)"}

        with open('gpu_memory.json', "w+") as f:
            json.dump(json_content, f)

        time.sleep(60 - (time.time() - t0))


# ------------ Training Sweep Functions------------ 
# - iterates across big_batch_sizes to get metrics for both regression and classif target models
  
def run_experiment_for_batch_size(x_train, y_train, x_test, y_test, il_model_loss_dict, small_batch_size, big_batch_size, epochs):
    ''' 
    Train 4 conv models and return training and test metrics    
            
    Train a conv model at the given big batch size for:
    - True and False values for null_hypothesis (meaning random subsampling or prioritized training subsampling)
    - Regression and classification loss for the target model  

    Subsampling rate is small_batch_size / big_batch_size  
    Holdout model is the same for all models (dataset, hyperparams, loss**)
    --- TODO test what happens when you have a regression holdout model instead of classification

    Return dicts of training/validation metrics
    '''

    t0 = time.time()
    nh_classif_metrics        = train_prioritized_conv(x_train, y_train, x_test, y_test, il_model_loss_dict, small_batch_size, big_batch_size, epochs, null_hypothesis=True , target_regression=False)
    classif_metrics           = train_prioritized_conv(x_train, y_train, x_test, y_test, il_model_loss_dict, small_batch_size, big_batch_size, epochs, null_hypothesis=False, target_regression=False)
    nh_regression_metrics     = train_prioritized_conv(x_train, y_train, x_test, y_test, il_model_loss_dict, small_batch_size, big_batch_size, epochs, null_hypothesis=True , target_regression=True)
    regression_metrics        = train_prioritized_conv(x_train, y_train, x_test, y_test, il_model_loss_dict, small_batch_size, big_batch_size, epochs, null_hypothesis=False, target_regression=True)
    
    t1 = time.time()
    print('took {} seconds for {}'.format(t1 - t0, big_batch_size))


    return nh_classif_metrics, classif_metrics, nh_regression_metrics, regression_metrics


def train_prioritized_conv(x_train, y_train, x_test, y_test, il_model_loss_dict, small_batch_size, big_batch_size, epochs, null_hypothesis=False, target_regression=False):
    ''' 
    Create a model with mse if target_regression is True, categorical_cross entropy if False
    Train a network given:
        - The big_batch_size
        - Whether the null hypothesis var is True
    
    '''

    t0 = time.time()
    model = ConvModel().create_model()
    if target_regression:
        model   = compile_model(model, loss='mse')
    else:
        model   = compile_model(model, loss='categorical_crossentropy')


    train_accuracies, train_losses, train_mse_list, train_cce_list = [], [], [], []
    val_accuracies  , val_losses  , val_mse       , val_cce        = [], [], [], []

    for epoch in range(epochs):
        print('\nepoch {}'.format(epoch))
        batch_train_accuracies, batch_train_losses, batch_train_mse, batch_train_cce = [], [], [], []
        image_label_dict_shuffled = shuffle_train_set_with_idx(x_train, y_train)

        pool = ThreadPool(processes=1)
        q = Queue(maxsize=1)
        
        i = 0
        while i + big_batch_size <= len(image_label_dict_shuffled):
            if i == 0:
                async_result = pool.apply_async(load_batch_into_queue, (model, image_label_dict_shuffled, q, i, big_batch_size, null_hypothesis))
                x_batch, y_batch = async_result.get().get() # get() results of load_batch_into_queue (a queue), and get() batch from queue
                async_result = pool.apply_async(load_batch_into_queue, (model, image_label_dict_shuffled, q, i + big_batch_size, big_batch_size, null_hypothesis))
                hist = model.fit(x_batch, y_batch, epochs=1, steps_per_epoch=small_batch_size, verbose=1, shuffle=False)
                x_next, y_next  = async_result.get().get() # first get retrieves result (a queue) from load_batch_into_queue, 2nd gets items from queue
            else:
                x_batch = x_next
                y_batch = y_next
                async_result = pool.apply_async(load_batch_into_queue, (model, image_label_dict_shuffled, q, i, big_batch_size, null_hypothesis))
            
                hist = model.fit(x_batch, y_batch, epochs=1, steps_per_epoch=small_batch_size, verbose=1, shuffle=False)

                x_next, y_next = async_result.get().get()
            
            # idxs               = [i[0]      for i in list(image_label_dict_shuffled.items())[i : i + big_batch_size]]
            # x_batch_candidates = [i[1]['x'] for i in list(image_label_dict_shuffled.items())[i : i + big_batch_size]]
            # y_batch_candidates = [i[1]['y'] for i in list(image_label_dict_shuffled.items())[i : i + big_batch_size]]

            # if null_hypothesis:
            #     x_batch = np.array([image_label_dict_shuffled[x]['x'] for x in range(i, i+small_batch_size)])
            #     y_batch = np.array([image_label_dict_shuffled[x]['y'] for x in range(i, i+small_batch_size)])
            # else:
            #     # # Evaluate the current model against the candidate batch and make a dict of frame idx: loss value
            #     #cur_candidate_loss_dict = eval_cur_model_on_batch(conv_model, x_batch_candidates, y_batch_candidates, idxs)
            #     preds = model.predict(np.array(x_batch_candidates), batch_size=small_batch_size, use_multiprocessing=True)
            #     loss_vals = categorical_crossentropy(preds, y_batch_candidates).numpy()
            #     cur_model_batch_loss_dict = {}
            #     for idx, loss_val in list(zip(idxs, loss_vals)):
            #         cur_model_batch_loss_dict[idx] = loss_val
            
            #     # Calculate RHO loss using the frame indices in both training loss dict and IL loss dict
            #     rho_loss = {}
            #     for frame_idx in cur_model_batch_loss_dict.keys():
            #         rho_loss[frame_idx] = cur_model_batch_loss_dict[frame_idx] - il_model_loss_dict[frame_idx]

            #     # Select the training batch using using the frames with highest RHO loss
            #     rho_loss_sorted = sorted(rho_loss.items(), key=lambda x: x[1])
            #     frame_idxs = [x[0] for x in rho_loss_sorted]

            #     # Select the training batch using the indices of the ideal frames from the subbatch
            #     x_batch = np.array([image_label_dict_shuffled[x]['x'] for x in frame_idxs])
            #     y_batch = np.array([image_label_dict_shuffled[x]['y'] for x in frame_idxs])


            # Shuffle is false because we already shuffled the data
            hist = model.fit(x_batch, y_batch, epochs=1, steps_per_epoch=small_batch_size, verbose=1, shuffle=False)
            batch_train_accuracies.append(hist.history['accuracy'][0])
            batch_train_losses    .append(hist.history['loss'][0])
            batch_train_mse       .append(hist.history['mse'][0])
            batch_train_cce       .append(hist.history['categorical_crossentropy'][0])

            i += big_batch_size
        
        train_accuracies.append(statistics.mean(batch_train_accuracies))
        print('avg accuracy over epoch: {:.2f}'.format(statistics.mean(batch_train_accuracies)))
        train_losses.    append(statistics.mean(batch_train_losses))
        print('avg loss over epoch: {:.2f}'    .format(statistics.mean(batch_train_losses)))
        train_mse_list.  append(statistics.mean(batch_train_mse))
        print('avg mse over epoch: {}'     .format(statistics.mean(batch_train_mse)))
        train_cce_list.  append(statistics.mean(batch_train_cce))
        print('avg cce over epoch: {}'     .format(statistics.mean(batch_train_cce)))

        val_hist = model.evaluate(x_test, y_test, return_dict=True)
        val_accuracies.append(val_hist['accuracy'])
        val_losses    .append(val_hist['loss'])
        val_mse       .append(val_hist['mse'])
        val_cce       .append(val_hist['categorical_crossentropy'])
  
    t1 = time.time()
    train_time = t1 - t0
    #print('took {} seconds'.format(t1 - t0))
    train_val_dict = {}
    train_val_dict['val'] = {'accuracy': val_accuracies , 
                                'loss': val_losses, 
                                'mse': val_mse,
                                'cce': val_cce,
                                'time': train_time}

    train_val_dict['train']   = {'accuracy':train_accuracies ,
                                'loss':train_losses ,
                                'mse': train_mse_list,
                                'cce': train_cce_list,
                                'time': train_time}

    return train_val_dict



if __name__ == '__main__':
    ''' 
    big_batch_sizes: List of big_batch_sizes to try, with larger values resulting in a smaller number of samples per epoch
    teacher_regression: If True, use regression loss (mse) on holdout model. If False, use a classification loss (categorical_crossentropy) 
    
    The function runs roughly as follows
    - 1) train a model on holdout dataset with mse or cce loss
    - 2) evaluate the holdout model against the *training* set and make a dict of frame_idx: loss value
    - 3) for each size in big_batch_sizes:
            - train 4 target models:
              - train a cce network with null hypothesis & another with prioritized training (classif target)
              - train a mse network with null hypothesis & another with prioritized training (regression target)

    '''
    
    t0 = time.time()

    log_util_thread = threading.Thread(target=log_util_usage) # , args=(args,))
    log_util_thread.daemon = True
    log_util_thread.start()


    # Load and prepare the data
    download_qmnist()
    qmnist = unpickle('qmnist-the-extended-mnist-dataset-120k-images/MNIST-120k')
    images = qmnist['data']
    labels = qmnist['labels']
    num_classes = len(set(labels.flatten()))
    (x_train, y_train), (x_test, y_test), (x_holdout, y_holdout) = prep_data(images, labels, num_classes=10)
    
    # big_batch_sizes = [10000, 5000, 2560, 1280, 640, 448]
    big_batch_sizes = [10000]#, 128, 72]
    small_batch_size = 64
    epochs = 10
    
    # Step 1: Train the IL model on the holdout data with a classification loss
    holdout_model = ConvModel()
    holdout_model = holdout_model.create_model()
    holdout_model = compile_model(holdout_model, loss='categorical_crossentropy')
    print('\n *** Fitting Holdout Model ***\n')
    fit(holdout_model, x_holdout, y_holdout, batch_size = small_batch_size, epochs=epochs, verbose=1, shuffle=True)
    
    # Step 2: Get Irreducible losses for the holdout model on the training set --> do __before__ shuffling data
    il_model_loss_dict        = make_il_loss_dict(holdout_model, x_train, y_train, loss='categorical_crossentropy')
    
    null_hypot_metrics_regression               = {}
    prioritized_training_metrics_regression     = {}
    null_hypot_metrics_classification           = {}
    prioritized_training_metrics_classification = {}

    for big_batch_size in big_batch_sizes:
        print('big_batch_size is {}'.format(big_batch_size))
        nh_reg_metrics, reg_metrics, nh_classif_metrics, classif_metrics  = run_experiment_for_batch_size(
            x_train, y_train, x_test, y_test, il_model_loss_dict, small_batch_size, big_batch_size, epochs=epochs)

        null_hypot_metrics_regression              [big_batch_size] = nh_reg_metrics
        prioritized_training_metrics_regression    [big_batch_size] = reg_metrics
        null_hypot_metrics_classification          [big_batch_size] = nh_classif_metrics
        prioritized_training_metrics_classification[big_batch_size] = classif_metrics

    dataset_fn = 'data-{}-epochs.json'.format(epochs)
    with open(dataset_fn, 'w', encoding='utf-8') as f:
        json.dump([{'Null Hypothesis Regression'         : null_hypot_metrics_regression}, 
                  {'Prioritized Training Regression'    : prioritized_training_metrics_regression},
                  {'Null Hypothesis Classification'     : null_hypot_metrics_classification},
                  {'Prioritized Training Classification': prioritized_training_metrics_classification}],
                f, ensure_ascii=False, indent=4)

        # json.dump([{'Null Hypothesis Regression': null_hypot_metrics_regression}, {'Prioritized Training Regression'    : prioritized_training_metrics_regression}], f, ensure_ascii=False, indent=4)

        # json.dump({'Null Hypothesis Regression'         : null_hypot_metrics_regression}, f, ensure_ascii=False, indent=4)
        # json.dump({'Prioritized Training Regression'    : prioritized_training_metrics_regression}, f, ensure_ascii=False, indent=4)
        # json.dump({'Null Hypothesis Classification'     : null_hypot_metrics_classification}, f, ensure_ascii=False, indent=4)
        # json.dump({'Prioritized Training Classification': prioritized_training_metrics_classification}, f, ensure_ascii=False, indent=4)

    t1 = time.time()
    print('took {} in total to run over {} big_batch_sizes'.format(str(datetime.time(t1 - t0)), len(big_batch_size)))

    import pdb; pdb.set_trace()

    
    
    