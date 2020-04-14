# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

### SET THESE PATHS MANUALLY #####################################################
# Full paths are required because otherwise the code will not know where to look
# when it is executed on one of the clusters.

at_biwi = False  # Are you running this code from the ETH Computer Vision Lab (Biwi)?
project_root = 'D:\\DeepLearning\\acdc_segmenterPM'
#data_root = 'D:\\Data\\Hakim\\ACDC\\training'
data_root = 'D:\\DeepLearning\Data\\NIIGZWithPM4D'
test_data_root = 'D:\\DeepLearning\Data\\testing'
local_hostnames = ['6QVR25']  # used to check if on cluster or not,
                                # enter the name of your local machine

##################################################################################

log_root = os.path.join(project_root, 'acdc_logdir')
preproc_folder = os.path.join(project_root,'preproc_data')

def setup_GPU_environment():

    if at_biwi:
        hostname = socket.gethostname()
        print('Running on %s' % hostname)
        if not hostname in local_hostnames:
            logging.info('Setting CUDA_VISIBLE_DEVICES variable...')
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
            logging.info('SGE_GPU is %s' % os.environ['SGE_GPU'])
    else:
        logging.warning('!! No GPU setup defined. Perhaps you need to set CUDA_VISIBLE_DEVICES etc...?')
