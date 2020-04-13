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

import layersProd as layers
import modelProd as model
import functions

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
alpha=10

def GetArea(mask):
	nx, ny = mask.shape
	area=0
	for x in range (0, nx):
		for y in range (0,ny):
			if mask[x,y]==1:
				area=area+1
	return area
def PrepareImages2D(img3D,pixel_size,normaliseBool,order):
    image_size = (212, 212)
    target_resolution = (1.36719, 1.36719)
    nx, ny = image_size[:2]

    batch_size=1

    if normaliseBool==True:
        img = functions.normalise_image(img3D)
    else:
        img = img3D

    scale_vector = (pixel_size[0] / target_resolution[0],pixel_size[1] / target_resolution[1])
    Images2D = []
        
    for zz in range(img.shape[2]):
                
                slice_img = np.squeeze(img[:,:,zz])
                slice_rescaled = transform.rescale(slice_img,
                                                   scale_vector,
                                                   order=order,
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

                Images2D.append(slice_cropped)
    return Images2D


def SplitTo3D(img_dat):
	
	Images3D = []
        
	for tt in range(img_dat.shape[3]):
         Images3D.append(img_dat[:,:,:,tt]) 

	return Images3D
    
def SplitTo2D(img_dat):
	
	Images2D = []
        
	for zz in range(img_dat.shape[2]):
         Images2D.append(img_dat[:,:,zz]) 

	return Images2D

def Recon3D(slices):	
    Images3D = (np.asarray(slices, dtype=np.uint8)).transpose(1,2,0)
    return Images3D

def convert4Dto3D(img4D):
    Images3D = SplitTo3D(img4D)
    listImages2D = []
    for i in range(len(Images3D)):
        Images2D = SplitTo2D(Images3D[i])
        for j in range (len(Images2D)):
            listImages2D.append(Images2D[j])

    Image3D = Recon3D(listImages2D)
    return Image3D
         
def unet2DvT2(image_file,output_file, checkpoint_path, do_postprocessing=False, use_iter=None):
        
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
    logging.info(' ----- Doing image4D: -------------------------')
    logging.info('Doing: %s' % image_file)
    logging.info(' --------------------------------------------')
    
    with tf.Session() as sess:

        sess.run(init)
        saver.restore(sess, checkpoint_path)

        total_time = 0
        total_volumes = 0

        logging.info(' ----- Doing image4D: -------------------------')
        logging.info('Doing: %s' % image_file)
        logging.info(' --------------------------------------------')

        file_base = image_file.split('.nii.gz')[0]
        print("file_base : " + file_base)
            
        img_dat = functions.load_nii(image_file)
        img4D = img_dat[0].copy()

        listImg=[]
        is4D = len(img4D.shape)==4
        if (not is4D):
           listImg.append(img_dat[0].copy())
           img4D=np.transpose(np.asarray(listImg, dtype=np.uint8), (1,2,3,0))
            
        print("img4D.shape[3] : " + str(img4D.shape[3]))			
            
        prediction4D = []						
        start_time = time.time()
       # Uncertainties=[]
        output_fileConf=output_file.replace(".nii.gz","Conf.txt")
        file = open(output_fileConf,"w") 
        for tt in range(img4D.shape[3]): 
            
            img = img4D[:,:,:,tt]
            img = functions.normalise_image(img)

            pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
            Nimg= PrepareImages2D(img4D[:,:,:,tt],pixel_size,0,1)
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
                TotalSize=GetArea(Myocardium[0])
                Certain=0
                meanT2=0
                if(TotalSize>0):
                   Probabilities=logits_out[:,:,:,1]*Myocardium
                   CertaintySlice=np.sum(Probabilities)
                   meanT2Slice=np.sum(Nimg[zz]*Myocardium)
                   
                   Certain=CertaintySlice/TotalSize
                   meanT2=meanT2Slice/TotalSize
                   print(Certain, meanT2)
                Uncertainties=[zz/img.shape[2],meanT2,Certain]
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
                scale_vector2 = (1.0/scale_vector[0], 1.0/scale_vector[1],1)
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
            prediction4D.append(prediction_arr)
            
        prediction4D_arr = np.transpose(np.asarray(prediction4D, dtype=np.uint8), (1,2,3,0))
        #prediction4D_arr = np.asarray(prediction4D, dtype=np.uint8)
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        total_volumes += 1

        logging.info('Evaluation of volume took %f secs.' % elapsed_time)

        # Save prediced mask
        out_affine = img_dat[1]
        out_header = img_dat[2]

        logging.info('saving to: %s' % output_file)
        if (not is4D):
            predictionIn3D = convert4Dto3D(prediction4D_arr)
            functions.save_nii(output_file, predictionIn3D, out_affine, out_header)
        else:
            functions.save_nii(output_file, prediction4D_arr, out_affine, out_header)

        file.close() 
        logging.info('Average time per volume: %f' % (total_time/total_volumes))

def selectNetRF(image4D,prediction4D,Select_out):
    img4D = image4D[0].copy()
            
    print(len(img4D.shape))
    listImg=[]
    is4D = len(img4D.shape)==4

    if (not is4D):
        listImg.append(image4D[0].copy())
        img4D=np.transpose(np.asarray(listImg, dtype=np.uint8), (1,2,3,0))
		   
    mask4D = prediction4D[0].copy()
    listMask=[]
    if (len(mask4D.shape)<4):
    	listMask.append(prediction4D[0].copy())
    	mask4D=np.transpose(np.asarray(listMask, dtype=np.uint8), (1,2,3,0))
 
    nbFrames = img4D.shape[3]

    list4D=[]
    for tt in range(nbFrames): 
            class_out=Select_out
            print(class_out)
            
            apexSlice=len(class_out)
            basalSlice=0
            pred2D = SplitTo2D(mask4D[:,:,:,tt])
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

                    areaEndo=GetArea(EndoPred)
                    areaEpi=GetArea(EpiPred)
                    if(areaEpi>3):
                        slice_img = EpiPred					
                        EpiPred=functions.getConvexHull(slice_img)					
                    if(areaEndo>3):
                        slice_img2 = EndoPred					
                        EndoPred=functions.getConvexHull(slice_img2)
                    KeepEpi=EpiPred*2
                    
                    if ((zz>=basalSlice) & (zz< basalSlice+2)):												
                    
                    ## 2. remove extra contouring by doing epi= dilated endo with fixed radius
                        dilatationRadius=math.sqrt(GetArea(EpiPred)/math.pi)-math.sqrt(GetArea(EndoPred)/math.pi)+1
                        ss=disk(int(dilatationRadius))
                        Correctionmask=morphology.binary_dilation(EndoPred,ss)
                        KeepEpi=(EpiPred+Correctionmask==2)*2
                    
                    if (zz==basalSlice):								
                        EndoPred=morphology.binary_dilation(EndoPred,disk(1))
                        
                    SlicesPred.append(KeepEpi+EndoPred)	
                else: 
                    SlicesPred.append(null2Dimage)

            recon3d = Recon3D(SlicesPred)
            list4D.append(recon3d)
    recon4d = np.transpose(np.asarray(list4D, dtype=np.uint8), (1,2,3,0))
    
    if (not is4D):
        return convert4Dto3D(recon4d)

    return recon4d
    	         
def callSelectorRF(image_file,prediction_file,Conf_file,modelSelection_path,output_file):
    prediction4D=functions.load_nii(prediction_file)
    img_dat = functions.load_nii(image_file)
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
	
    Result = selectNetRF(img_dat,prediction4D,Select_out)
    logging.info('Editing the prediction with the slices selected')

    logging.info('Selected segmentation save to : %s' %output_file)
    print('RESULT SHAPE : ' + str(Result.shape))
    functions.save_nii(output_file, Result, prediction4D[1], prediction4D[2])
                        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script to process data of a specific project")
    parser.add_argument('-image', '--image_file', type=str,help='(Single case mode) path to 3D image')
    args = parser.parse_args()
    
    model_path = os.path.abspath("Models/T2/")

    modelSliceSelectionT2_path = os.path.join(model_path, 'T2_RF_model.sav')
    modelSegmentaion_path = os.path.join(model_path, 'T2SegmentationModel.ckpt')
    image_file = args.image_file

    if image_file is None:
        logging.warning('image_file should be set')
    else:
        #######################
        image_size = (212, 212)
        batch_size = 1
        num_channels = 3
        nx, ny = image_size[:2]
        
        #######################
        if image_file is not None:
            logging.warning('Processing image : ' + image_file)
            if os.path.exists(image_file):
                logging.info("T2 Image3D found")
                prediction_file = image_file.replace(".nii.gz","Seg.nii.gz")
                Conf_file=image_file.replace(".nii.gz","SegConf.txt")
                if os.path.exists(prediction_file):
                    logging.info("Prediction found -> We don't need to perform a new segmentation for "+ image_file)
                else: 
                    tf.reset_default_graph()				
                    unet2DvT2(image_file,prediction_file, modelSegmentaion_path, do_postprocessing=True,   use_iter=None)
                    logging.info("Image processed : " + image_file)            
                predictionSelected_file = prediction_file.replace(".nii.gz","Selected.nii.gz")
                if os.path.exists(predictionSelected_file):
                    logging.info("Prediction Selected found -> We don't need to perform a new selection for "+ prediction_file)
                else:
                    callSelectorRF(image_file,prediction_file,Conf_file,modelSliceSelectionT2_path,predictionSelected_file)