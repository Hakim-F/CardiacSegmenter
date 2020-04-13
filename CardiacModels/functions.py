# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import numpy as np
from skimage import measure
from skimage import transform
import logging
import nibabel as nib
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

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

    if normaliseBool==True:
        img = normalise_image(img3D)
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

def normalise_image(image):
    '''
    make image zero mean and unit standard deviation
    '''

    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s)

def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''

    out_img = np.zeros(mask.shape, dtype=np.uint8)

    for struc_id in [1, 2, 3]:

        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)

        props = measure.regionprops(blobs)

        if not props:
            continue

        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label

        out_img[blobs == largest_blob_label] = struc_id

    return out_img
    
def keep_largest_connected_componentsVersionPM(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''

    out_img = np.zeros(mask.shape, dtype=np.uint8)

    for struc_id in [1, 2, 3, 4]: 

        if struc_id != 4:
            binary_img = mask == struc_id
            blobs = measure.label(binary_img, connectivity=1)

            props = measure.regionprops(blobs)

            if not props:
                continue

            area = [ele.area for ele in props]
            largest_blob_ind = np.argmax(area)
            largest_blob_label = props[largest_blob_ind].label

            out_img[blobs == largest_blob_label] = struc_id
        else:
            out_img[mask==struc_id]= struc_id

    return out_img

def keep_largest_connected_componentsVersionScar(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''

    out_img = np.zeros(mask.shape, dtype=np.uint8)

    for struc_id in [1, 2, 3]: 

        if struc_id != 3:
            binary_img = mask == struc_id
            blobs = measure.label(binary_img, connectivity=1)

            props = measure.regionprops(blobs)

            if not props:
                continue

            area = [ele.area for ele in props]
            largest_blob_ind = np.argmax(area)
            largest_blob_label = props[largest_blob_ind].label

            out_img[blobs == largest_blob_label] = struc_id
        else:
            out_img[mask==struc_id]= struc_id

    return out_img

def load_nii(img_path):

    '''
    Shortcut to load a nifti file
    '''

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

def save_nii(img_path, data, affine, header):
    '''
    Shortcut to save a nifty file
    '''

    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)

def getConvexHull(data):
    region = np.argwhere(data)
    hull = ConvexHull(region)
    verts = [(region[v,0], region[v,1]) for v in hull.vertices]
    img = Image.new('L', data.shape, 0)
    ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
    mask = np.array(img)

    return mask.T