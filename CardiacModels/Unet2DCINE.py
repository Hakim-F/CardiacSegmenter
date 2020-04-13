import os
import numpy as np
import logging

import argparse
import time
import tensorflow as tf
from skimage import transform

import modelProd as model
import functions


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def unet2D(image_file,output_file, checkpoint_path, do_postprocessing=False, use_iter=None):
    
    image_size = (212, 212)
    batch_size = 1
    num_channels = 4
    target_resolution = (1.36719, 1.36719)
    nx, ny = image_size[:2]
    
    image_tensor_shape = [batch_size] + list(image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
    mask_pl, softmax_pl = model.predict(images_pl,num_channels)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        saver.restore(sess, checkpoint_path)

        total_time = 0
        total_volumes = 0

        file_base = image_file.split('.nii.gz')[0]
        print("file_base : " + file_base)
            
        img_dat = functions.load_nii(image_file)
        img4D = img_dat[0].copy()
        if (len(img4D.shape)!=4):
            print('ERROR the input image needs to be 4D')
            return False

        print("img4D.shape[3] : " + str(img4D.shape[3]))			
            
        prediction4D = []						
        start_time = time.time()
        for tt in range(img4D.shape[3]): 
            
            img = img4D[:,:,:,tt]
            img = functions.normalise_image(img)
                
            pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
            scale_vector = (pixel_size[0] / target_resolution[0],
                            pixel_size[1] / target_resolution[1])

            predictions = []
            
            for zz in range(img.shape[2]):
                
                slice_img = np.squeeze(img[:,:,zz])
                slice_rescaled = transform.rescale(slice_img,
                                                   scale_vector,
                                                   order=1,
                                                   preserve_range=True,
                                                   mode='constant')

                x, y = slice_rescaled.shape

                x_s = (x - nx) // 2
                y_s = (y - ny) // 2
                x_c = (nx - x) // 2
                y_c = (ny - y) // 2

                # Crop section of image for prediction
                if x > nx and y > ny:
                    slice_cropped = slice_rescaled[x_s:x_s+nx, y_s:y_s+ny]
                else:
                    slice_cropped = np.zeros((nx,ny))
                    if x <= nx and y > ny:
                        slice_cropped[x_c:x_c+ x, :] = slice_rescaled[:,y_s:y_s + ny]
                    elif x > nx and y <= ny:
                        slice_cropped[:, y_c:y_c + y] = slice_rescaled[x_s:x_s + nx, :]
                    else:
                        slice_cropped[x_c:x_c+x, y_c:y_c + y] = slice_rescaled[:, :]


                # GET PREDICTION
                network_input = np.float32(np.tile(np.reshape(slice_cropped, (nx, ny, 1)), (batch_size, 1, 1, 1)))
                mask_out, logits_out = sess.run([mask_pl, softmax_pl], feed_dict={images_pl: network_input})
                prediction_cropped = np.squeeze(logits_out[0,...])

                # ASSEMBLE BACK THE SLICES
                slice_predictions = np.zeros((x,y,num_channels))
                # insert cropped region into original image again
                if x > nx and y > ny:
                    slice_predictions[x_s:x_s+nx, y_s:y_s+ny,:] = prediction_cropped
                else:
                    if x <= nx and y > ny:
                        slice_predictions[:, y_s:y_s+ny,:] = prediction_cropped[x_c:x_c+ x, :,:]
                    elif x > nx and y <= ny:
                        slice_predictions[x_s:x_s + nx, :,:] = prediction_cropped[:, y_c:y_c + y,:]
                    else:
                        slice_predictions[:, :,:] = prediction_cropped[x_c:x_c+ x, y_c:y_c + y,:]

                # RESCALING ON THE LOGITS
                scale_vector2 = (1.0/scale_vector[0],1.0/scale_vector[1])
                prediction = transform.rescale(slice_predictions,
                                                scale_vector2,
                                                order=1,
                                                preserve_range=True,
                                                mode='constant')

                prediction = np.uint8(np.argmax(prediction, axis=-1))
                predictions.append(prediction)

            prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))
            
            # This is the same for 2D and 3D again
            if do_postprocessing:
                prediction_arr = functions.keep_largest_connected_components(prediction_arr)
            prediction4D.append(prediction_arr)
            
        prediction4D_arr = np.transpose(np.asarray(prediction4D, dtype=np.uint8), (1,2,3,0))
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        total_volumes += 1

        logging.info('Evaluation of volume took %f secs.' % elapsed_time)

        # Save prediced mask
        out_affine = img_dat[1]
        out_header = img_dat[2]

        logging.info('saving to: %s' % output_file)
        functions.save_nii(output_file, prediction4D_arr, out_affine, out_header)
     
        logging.info('Average time per volume: %f' % (total_time/total_volumes))
        return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script to run deep learning model on CINE image (nii.gz 4D)")
    parser.add_argument('-image', '--image_file', type=str,help='(Single case mode) path to 4D image')
    args = parser.parse_args()
    
    model_path = os.path.abspath("Models/CINE/")
    modelCine_path = os.path.join(model_path, 'CINEModel.ckpt')
    
    image_file = args.image_file

    if image_file is None:
        logging.warning('image_file should be set')
    elif os.path.exists(image_file):
        prediction_file = image_file.replace(".nii.gz","Seg.nii.gz")
        if os.path.exists(prediction_file):
            logging.info("Prediction found -> We don't need to perform a new segmentation for "+ image_file)
        else:            
            logging.info("Segmentation Model found : " + modelCine_path)
            logging.warning('Starting inference on image : ' + image_file)
            if (unet2D(image_file,prediction_file,
                                    modelCine_path,
                                    do_postprocessing=True,
                                    use_iter=None)):
                logging.info("Segmentation mask created : "+ prediction_file)