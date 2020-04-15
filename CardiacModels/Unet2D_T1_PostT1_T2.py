import os
import numpy as np
import logging

import argparse
import time
import tensorflow as tf
from skimage import transform
from skimage import morphology
from skimage.morphology import disk
import math

from sklearn.ensemble import RandomForestClassifier
import pickle
import modelProd as model
import functions

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def unet2D(image_file,output_file, checkpoint_path, do_postprocessing=False, use_iter=None):
        
    image_size = (212, 212)
    batch_size = 1
    num_channels = 3
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
        img3D = img_dat[0].copy()
        print("image shape : " + str(img3D.shape))
        is3D = len(img3D.shape)==3

        print("is the image 3D ? : " + str(is3D))	
        if  (not is3D):
            print("ERROR : The image needs to be 3D ")
            return False		
					
        start_time = time.time()
       # Uncertainties=[]
        output_fileConf=output_file.replace(".nii.gz","Conf.txt")
        file = open(output_fileConf,"w") 
    
        img = functions.normalise_image(img3D)
        
        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
        Nimg= functions.PrepareImages2D(img3D,pixel_size,0,1)
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

            
            network_input = np.float32(np.tile(np.reshape(slice_cropped, (nx, ny, 1)), (batch_size, 1, 1, 1)))
            mask_out, logits_out = sess.run([mask_pl, softmax_pl], feed_dict={images_pl: network_input})
            
################## Measure uncertainties based on softmax prediction=logits_out ################
            Myocardium=(mask_out==1)*1
            TotalSize=functions.GetArea(Myocardium[0])
            Certain=0
            MeanValue=0
            if(TotalSize>0):
                Probabilities=logits_out[:,:,:,1]*Myocardium
                CertaintySlice=np.sum(Probabilities)
                MeanValueSlice=np.sum(Nimg[zz]*Myocardium)
                
                Certain=CertaintySlice/TotalSize
                MeanValue=MeanValueSlice/TotalSize
                print(Certain, MeanValue)
            Uncertainties=[zz/img.shape[2],MeanValue,Certain]
            file.write(" ".join(str(elem) for elem in Uncertainties) + "\n")
################### create output mask ########################
            
            prediction_cropped = np.squeeze(logits_out[0,...])

            # ASSEMBLE BACK THE SLICES
            slice_predictions = np.zeros((x,y,num_channels))
            # insert cropped region into original image again
            if x > nx and y > ny:
                #print("prediction_cropped.shape : " + str(prediction_cropped.shape))
                slice_predictions[x_s:x_s+nx, y_s:y_s+ny,:] = prediction_cropped
            else:
                if x <= nx and y > ny:
                    slice_predictions[:, y_s:y_s+ny,:] = prediction_cropped[x_c:x_c+ x, :,:]
                elif x > nx and y <= ny:
                    slice_predictions[x_s:x_s + nx, :,:] = prediction_cropped[:, y_c:y_c + y,:]
                else:
                    slice_predictions[:, :,:] = prediction_cropped[x_c:x_c+ x, y_c:y_c + y,:]

            # RESCALING ON THE LOGITS
            scale_vector2 = (1.0/scale_vector[0], 1.0/scale_vector[1])
            prediction = transform.rescale(slice_predictions,
                                            scale_vector2,
                                            order=1,
                                            preserve_range=True,
                                            mode='constant')

            prediction = np.uint8(np.argmax(prediction, axis=-1))
            predictions.append(prediction)

        prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))
        prediction_arr[prediction_arr>=1] = prediction_arr[prediction_arr>=1]+1
        # This is the same for 2D and 3D again
        if do_postprocessing:
            prediction_arr = functions.keep_largest_connected_components(prediction_arr)

        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        total_volumes += 1

        logging.info('Evaluation of volume took %f secs.' % elapsed_time)

        # Save prediced mask
        out_affine = img_dat[1]
        out_header = img_dat[2]

        logging.info('saving to: %s' % output_file)

        functions.save_nii(output_file, prediction_arr, out_affine, out_header)

       # file.writelines(" ".join(str(elem) for elem in Uncertainties) + "\n")
		#["%s\n" % item  for item in Uncertainties])
        file.close() 
        logging.info('Average time per volume: %f' % (total_time/total_volumes))
    return True

def selectNetRF(prediction3D,Select_out):
    
    mask3D = prediction3D[0].copy()
    class_out=Select_out
    print(class_out)
    
    apexSlice=len(class_out)
    basalSlice=0
    pred2D = functions.SplitTo2D(mask3D)
    numberSlices=len(pred2D)
    print(numberSlices)
    for zz in range(int((numberSlices/3) +0.5)):
        if class_out[zz]==1:
            basalSlice=zz
            break
    for zz in range(int((numberSlices/3) +0.5)):
        if class_out[numberSlices-zz-1]==1:
            apexSlice=numberSlices-zz-1
            break

    print("basalSlice : " + str(basalSlice))
    print("apexSlice : " + str(apexSlice))
    
    null2Dimage=np.zeros(pred2D[0].shape)
    SlicesPred=[]
    for zz in range(numberSlices):
        if ((zz>=basalSlice) & (zz<=apexSlice)):
            EpiPred=(pred2D[zz]>=2)*1
            EndoPred=(pred2D[zz]==3)*1					

            areaEndo=functions.GetArea(EndoPred)
            areaEpi=functions.GetArea(EpiPred)
            if(areaEpi>3):
                slice_img = EpiPred					
                EpiPred=functions.getConvexHull(slice_img)					
            if(areaEndo>3):
                slice_img2 = EndoPred					
                EndoPred=functions.getConvexHull(slice_img2)
            KeepEpi=EpiPred*2
            
            if ((zz>=basalSlice) & (zz< basalSlice+2)):												
            
            ## 2. remove extra contouring by doing epi= dilated endo with fixed radius
                dilatationRadius=math.sqrt(functions.GetArea(EpiPred)/math.pi)-math.sqrt(functions.GetArea(EndoPred)/math.pi)+1
                ss=disk(int(dilatationRadius))
                Correctionmask=morphology.binary_dilation(EndoPred,ss)
                KeepEpi=(EpiPred+Correctionmask==2)*2
            
            if (zz==basalSlice):								
                EndoPred=morphology.binary_dilation(EndoPred,disk(1))
                
            SlicesPred.append(KeepEpi+EndoPred)	
        else: 
            SlicesPred.append(null2Dimage)

    recon3d = functions.Recon3D(SlicesPred)
    return recon3d

def callSelectorRF(prediction_file,Conf_file,modelSelection_path,output_file):
    prediction3D=functions.load_nii(prediction_file)
    logging.info('Selector called')
    Input=[]
    with open(Conf_file) as fd:
       for ln in fd:	
          Input.append([float(str.strip(x)) for x in ln.split(' ')])

    print(Input)
            
    logging.info('RF selection')
    SelectionModel = pickle.load(open(modelSelection_path, 'rb'))	
	
    Select_out=SelectionModel.predict(Input)
    print(Select_out)
	
    Result = selectNetRF(prediction3D,Select_out)
    logging.info('Editing the prediction with the slices selected')

    logging.info('Selected segmentation save to : %s' %output_file)

    functions.save_nii(output_file, Result, prediction3D[1], prediction3D[2])
                        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script to process data of a specific project")
    parser.add_argument('-image', '--image_file', type=str,help='(Single case mode) path to 3D image')
    parser.add_argument('-sequence', '--sequence', type=str, help='Choose a sequence among : T1, PostT1 or T2')
    args = parser.parse_args()
    
    sequence = args.sequence
    image_file = args.image_file

    if image_file is None:
        logging.warning('image_file should be set')
    elif sequence is None:
        logging.warning('The argument sequence should be given')
    elif sequence not in ["T1","T2","PostT1"]:
        logging.warning('The only sequences available are T1, PostT1 and T2')
    elif os.path.exists(image_file):
        prediction_file = image_file.replace(".nii.gz","Seg.nii.gz")
        logging.info(image_file + " found")
        if os.path.exists(prediction_file):
            logging.info("Prediction found -> We don't need to perform a new segmentation for "+ image_file)
        else:
            logging.info("Looking for Segmentation Model and RandomForest Selection Model")
            
            dir_path = os.path.dirname(os.path.realpath(__file__))
            model_path = os.path.join("Models/",sequence)
            model_path = os.path.join(dir_path,model_path)
            model_path = os.path.abspath(model_path)
            
            modelSliceSelection_path = os.path.join(model_path, 'T1_RF_model.sav')
            modelSegmentaion_path = os.path.join(model_path, 'PreT1SegmentationModel.ckpt')
            if sequence == "T1":
                modelSliceSelection_path = os.path.join(model_path, 'T1_RF_model.sav')
                modelSegmentaion_path = os.path.join(model_path, 'PreT1SegmentationModel.ckpt')
            if sequence == "PostT1":
                modelSliceSelection_path = os.path.join(model_path, 'PostT1_RF_model.sav')
                modelSegmentaion_path = os.path.join(model_path, 'PostT1SegmentationModel.ckpt')
            if sequence == "T2":
                modelSliceSelection_path = os.path.join(model_path, 'T2_RF_model.sav')
                modelSegmentaion_path = os.path.join(model_path, 'T2SegmentationModel.ckpt')

            logging.info("Segmentation Model found : " + modelSegmentaion_path)
            logging.info("RandomForest Selection Model found : " + modelSliceSelection_path)
            logging.warning('Starting inference on image : ' + image_file)
            
            Conf_file=image_file.replace(".nii.gz","SegConf.txt")
            tf.reset_default_graph()
            if (unet2D(image_file,prediction_file, modelSegmentaion_path, do_postprocessing=True,   use_iter=None)):
                logging.info("Segmentation mask created : "+ prediction_file)            
                predictionSelected_file = prediction_file.replace(".nii.gz","Selected.nii.gz")

                if os.path.exists(predictionSelected_file):
                    logging.info("Segmentation mask with slice selection exists already -> We don't need to perform a new selection for "+ prediction_file)
                else:
                    logging.info("Performing slice selection on segmentation mask created") 
                    callSelectorRF(prediction_file,Conf_file,modelSliceSelection_path,predictionSelected_file)
                    logging.info("Segmentation mask with slice selection saved as : " + predictionSelected_file)
    else:
        logging.info(image_file + " not found")