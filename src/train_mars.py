"""
Retrain the YOLO model for your own dataset.
"""

''' Only use a certain GPU number '''
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from model import preprocess_true_boxes, yolo_body, yolo_loss, create_model
from utils import get_random_yolo_data, get_random_domain_data
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)

#CHANGE USERPATH
USERPATH = '/Users/stefano/Desktop//SeniorProject/YOCO/'

def main():
    model_name = 'YOCOv0.1-ailMars'
    
    # Training data/variable setup
    box_anno_path = USERPATH + "model_data/annotations/box_anno_ailMars.txt"
    dom_anno_path = USERPATH + "model_data/annotations/dom_anno_ailMars.txt"
    classes_path = USERPATH + 'model_data/class_lists/ailMars_classes.txt'
    log_dir = USERPATH + 'logs/' + model_name  + '/'
    
    anchors_path = USERPATH + 'model_data/anchors/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    input_shape = (1024,1024) # multiple of 32, hw

    # Create YOCO model
    model = create_model(input_shape, anchors, num_classes,
        weights_path= USERPATH + 'model_data/yolov3.h5') # make sure you know what you freeze
    model.summary()

    # Callbacks
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(filepath=log_dir + model_name + '_ep{epoch:03d}.ckpt', monitor='yolo_loss', save_weights_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='yolo_loss', factor=0.1, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='yolo_loss', min_delta=0, patience=10, verbose=1)
    nan_term = tf.keras.callbacks.TerminateOnNaN()
    csv_logger = CSVLogger(log_dir + model_name + '_training_log.csv', append=True, separator=',')

    # Validation split
    val_split = 0.1
    with open(box_anno_path) as f:
        box_lines = f.readlines()
    with open(dom_anno_path) as f:
        dom_lines = f.readlines()
        
    np.random.seed(10101)
    np.random.shuffle(box_lines)
    np.random.shuffle(dom_lines)
    np.random.seed(None)
    num_box_val = int(len(box_lines)*val_split)
    num_box_train = len(box_lines) - num_box_val
    num_dom_val = int(len(dom_lines)*val_split)
    num_dom_train = len(dom_lines) - num_dom_val

    # Training/loss parameters
    batch_size  = 16
    start_epoch = 0
    # ----------------
    stage1_epochs = 1
    stage2_epochs = 2
    stage1_lr = 1e-3
    stage2_lr = 1e-4
    # ----------------
    img_weight = 1
    inst1_weight = 1
    inst2_weight = 1
    inst3_weight = 1
    
    print('\nYOLO Detection training on {} samples, val on {} samples'.format(num_box_train, num_box_val))
    print('Image/Instance Discriminators training on {} samples, val on {} samples\n'.format(num_dom_train, num_dom_val))
    print("--- TRAINING PARAMETERS ---")
    print("Batch size: {}\nStage1 Epochs: {}\nStage1 LR: {}\nStage2 Epochs: {}\nStage2 LR: {}\nimg Loss Weight: {}\ninst1/2/3 Loss Weights: {},{},{}\n----------\n".format(
                    batch_size, stage1_epochs, stage1_lr, stage2_epochs, stage2_lr, img_weight, inst1_weight, inst2_weight, inst3_weight))
    # ------------------------------------------------------------------------------------------------------------------------------
    
    # Train with frozen layers first, to get a stable loss. This step is enough to obtain a not bad model.
    model.compile(optimizer=Adam(learning_rate=stage1_lr), 
        loss={
            'yolo': lambda y_true, y_pred: y_pred,
            'img': tf.keras.losses.BinaryCrossentropy(from_logits=True),
            'inst1': tf.keras.losses.BinaryCrossentropy(from_logits=True),
            'inst2': tf.keras.losses.BinaryCrossentropy(from_logits=True),
            'inst3': tf.keras.losses.BinaryCrossentropy(from_logits=True),
        },
        loss_weights=[
            1,
            img_weight,
            inst1_weight,
            inst2_weight,
            inst3_weight
        ])
    model.fit(data_generator_wrapper(box_lines[:num_box_train], dom_lines[:num_dom_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_box_train//batch_size),
            validation_data=data_generator_wrapper(box_lines[num_box_train:], dom_lines[num_dom_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_box_val//batch_size),
            epochs=stage1_epochs,
            initial_epoch=start_epoch,
            callbacks=[logging, checkpoint, early_stopping, csv_logger, nan_term])
    model.save_weights(log_dir + model_name + '_trained_weights_stage1.h5')
        
    print("\nSTAGE 1 TRAINING COMPLETE\n")

# ------------------------------------
    # Unfreeze and continue training, to fine-tune. Train longer if the result is not good.
    print("Unfreezing whole model...")
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(learning_rate=stage2_lr), # recompile to apply the change
        loss={
            'yolo': lambda y_true, y_pred: y_pred,
            'img': tf.keras.losses.BinaryCrossentropy(from_logits=True),
            'inst1': tf.keras.losses.BinaryCrossentropy(from_logits=True),
            'inst2': tf.keras.losses.BinaryCrossentropy(from_logits=True),
            'inst3': tf.keras.losses.BinaryCrossentropy(from_logits=True),
        },
        loss_weights=[
            1,
            img_weight,
            inst1_weight,
            inst2_weight,
            inst3_weight
        ])
    model.fit(data_generator_wrapper(box_lines[:num_box_train], dom_lines[:num_dom_train], batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_box_train//batch_size),
        validation_data=data_generator_wrapper(box_lines[num_box_train:], dom_lines[num_dom_train:], batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_box_val//batch_size),
        epochs=(stage1_epochs + stage2_epochs),
        initial_epoch=stage1_epochs,
        callbacks=[logging, checkpoint, early_stopping, reduce_lr, csv_logger, nan_term])
    model.save_weights(log_dir + model_name + '_weights_stage2.ckpt')
    
    print("\nSTAGE 2 TRAINING COMPLETE\n")
# -----------------------------------------

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def data_generator(box_lines, dom_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    np.random.shuffle(dom_lines)
    # pick b/2 source lines : b/2 target lines
    src_lines = []
    tgt_lines = []
    for line in dom_lines:
        l = line.rstrip()
        dom = l.split(' ')[-1]
        if dom == '0' and len(src_lines) < batch_size/2:
            src_lines.append(line)
        elif dom == '1' and len(tgt_lines) < (batch_size/2)+1:
            tgt_lines.append(line)
        
        if len(src_lines) == batch_size/2 and len(tgt_lines) == batch_size/2:
            break
        
    dom_lines = src_lines + tgt_lines

    n = len(box_lines)
    i = 0
    while True:
        box_image_data = []
        box_data = []
        domain_image_data = []
        domain_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(box_lines)
            box_image, box = get_random_yolo_data(box_lines[i], input_shape, random=True)
            domain_image, domain = get_random_domain_data(dom_lines[b], input_shape, random=True)
            box_image_data.append(box_image)
            box_data.append(box)
            domain_image_data.append(domain_image)
            domain_data.append(domain)
            i = (i+1) % n
        box_image_data = np.array(box_image_data)
        box_data = np.array(box_data)
        domain_image_data = np.array(domain_image_data)
        domain_data = np.array(domain_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [box_image_data, *y_true, domain_image_data], [np.zeros(batch_size), domain_data, domain_data, domain_data, domain_data]

def data_generator_wrapper(box_lines, dom_lines, batch_size, input_shape, anchors, num_classes):
    n = len(box_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(box_lines, dom_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    main()
