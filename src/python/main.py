'''
Training and testing script for CompenNet++.

This script trains/tests CompenNet++ on different dataset specified in 'data_list' below.
The detailed training options are given in 'train_option' below.

1. We start by setting the training environment to GPU (if any), then put K=20 setups in 'data_list', the 20 setups are used in our paper.
2. We set number of training images to 500 and loss function to l1+ssim, you can add other num_train and loss to 'num_train_list' and 'loss_list' for
comparison. Other training options are specified in 'train_option'.
3. The training data 'train_data' and validation data 'valid_data', are loaded in RAM using function 'loadData', and then we train the model with
function 'trainModel'. The training and validation results are both updated in Visdom window (`http://server:8097`) and console.
4. Once the training is finished, we can compensate the desired image. The compensation images 'prj_cmp_test' can then be projected to the surface.

Example:
    python main.py

See CompenNetPlusplusModel.py for CompenNet++ structure.
See CompenNetPlusplusDataset.py for training and validation data loading.
See trainNetwork.py for detailed training process.
See utils.py for helper functions.
'''

from trainNetwork import *
import CompenNetPlusplusModel

# %% Set environment
# set which GPUs to use
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'
device_ids = [0, 1, 2]

# set PyTorch device to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() >= 1:
    print('Train with', torch.cuda.device_count(), 'GPUs!')
else:
    print('Train with CPU!')

# repeatability
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% K=24 setups
# dataset_root = fullfile(os.getcwd(), '../../Dataset')
#
# data_list = [
#     'canonCamlink/dark/pos1/cloud_np',
#     'canonCamlink/dark/pos1/lavender_np',
#     'canonCamlink/dark/pos2/cubes_np',
#     'canonCamlink/dark/pos3/bars_spec_np',
#     'canonCamlink/dark/pos3/bubbles_np',
#     'canonCamlink/dark/pos3/curves_np',
#     'canonCamlink/dark/pos3/lavender_spec_np',
#     'canonCamlink/dark/pos5/curves_np',
#     'canonCamlink/dark/pos5/lavender_np',
#     'canonCamlink/dark/pos5/stripes_np',
#     'canonCamlink/dark/pos6/cubes_np',
#     'canonCamlink/dark/pos6/curves_np',
#     'canonCamlink/dark/pos6/lavender_np',
#     'canonCamlink/dark/pos6/pillow_np',
#     'canonCamlink/dark/pos6/stripes_np',
#     'canonCamlink/light/pos4/bubbles_np',
#     'canonCamlink/light/pos4/cloud_np',
#     'canonCamlink/light/pos4/curves_np',
#     'canonCamlink/light/pos4/squares_np',
#     'canonCamlink/light/pos4/water_np',
# ]

dataset_root = fullfile(os.getcwd(), '../../public')
# dataset_root = fullfile('D:/iccv19_dataset/public')

data_list = [
    'light1/pos1/cloud_np',
    'light1/pos1/lavender_np',
    'light1/pos2/cubes_np',
    'light1/pos3/bars_spec_np',
    'light1/pos4/bubbles_np',
    'light1/pos5/pillow_np',
    'light2/pos1/curves_np',
    'light2/pos1/lavender_np',
    'light2/pos1/stripes_np',
    'light2/pos2/lavender_spec_np',
    'light2/pos3/curves_np',
    'light2/pos4/lavender_np',
    'light2/pos5/stripes_np',
    'light2/pos6/cubes_np',
    'light2/pos6/curves_np',
    'light3/pos1/bubbles_np',
    'light3/pos1/cloud_np',
    'light3/pos1/squares_np',
    'light3/pos2/curves_np',
    'light3/pos2/water_np',
]

# Training configurations of CompenNet++ reported in the paper
num_train_list = [500]
loss_list = ['l1+ssim']

# You can also compare different configurations, such as different number of training images and loss functions as shown below
# num_train_list = [48, 125, 250, 500]
# loss_list = ['l1', 'l2', 'ssim', 'l1+l2', 'l1+ssim', 'l2+ssim', 'l1+l2+ssim']

# You can create your own models in CompenNetPlusplusModel.py and put their names in this list for comparisons.
model_list = ['CompenNet++', 'CompenNet++_w/o_refine', 'CompenNet++_fast', 'CompenNet++_faster']

train_option_default = {'data_name': '',  # will be set later
                        'model_name': '',
                        'num_train': '',
                        'max_iters': 1500,  # reduced this to 1000/500 to train model CompenNet++ fast/faster
                        'batch_size': 48,  # reduced this to 24/16 to train model CompenNet++ fast/faster
                        'lr': 1e-3,  # learning rate
                        'lr_drop_ratio': 0.2,
                        'lr_drop_rate': 1000,  # adjust this according to max_iters (lr_drop_rate < max_iters)
                        'loss': '',  # loss will be set to one of the loss functions in loss_list later
                        'l2_reg': 1e-4,  # l2 regularization
                        'device': device,
                        'plot_on': True,  # plot training progress using visdom
                        'train_plot_rate': 50,  # training and visdom plot rate
                        'valid_rate': 200}  # validation and visdom plot rate

# a flag that decides whether to compute and save the compensated images to the drive
save_compensation = True

# log file
from time import localtime, strftime

log_dir = '../../log'
if not os.path.exists(log_dir): os.makedirs(log_dir)
log_file_name = strftime('%Y-%m-%d_%H_%M_%S', localtime()) + '.txt'
log_file = open(fullfile(log_dir, log_file_name), 'w')
title_str = '{:30s}{:<30}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}\n'
log_file.write(title_str.format('data_name', 'model_name', 'loss_function',
                                'num_train', 'batch_size', 'max_iters',
                                'uncmp_psnr', 'uncmp_rmse', 'uncmp_ssim',
                                'valid_psnr', 'valid_rmse', 'valid_ssim'))
log_file.close()

# resize the input images if input_size is not None
input_size = None
# input_size = (256, 256) # we can also use a low-res input to speed up training/testing with a sacrifice of precision

# initialize CompenNet by loading the weights if it exists, otherwise quickly pretrain CompenNet
compen_net = CompenNetPlusplusModel.CompenNet()
if torch.cuda.device_count() >= 1: compen_net = nn.DataParallel(compen_net, device_ids=device_ids).to(device)
compen_net = initCompenNet(compen_net, dataset_root, device)

# %% evaluate all K=20 setups
# stats for different setups
for data_name in data_list:
    # load training and validation data
    data_root = fullfile(dataset_root, data_name)
    cam_surf, cam_train, cam_valid, prj_train, prj_valid, mask_corners = loadData(dataset_root, data_name, input_size)

    # surface image for training and validation
    cam_surf_train = cam_surf.expand_as(cam_train)
    cam_surf_valid = cam_surf.expand_as(cam_valid)

    # convert valid data to CUDA tensor if you have sufficient GPU memory (significant speedup)
    cam_valid.to(device)
    prj_valid.to(device)

    # validation data, 200 image pairs
    valid_data = dict(cam_surf=cam_surf_valid, cam_valid=cam_valid, prj_valid=prj_valid)

    # stats for different #Train
    for num_train in num_train_list:
        train_option = train_option_default.copy()
        train_option['num_train'] = num_train

        # select a subset to train
        train_data = dict(cam_surf=cam_surf_train[:num_train, :, :, :], cam_train=cam_train[:num_train, :, :, :],
                          prj_train=prj_train[:num_train, :, :, :])

        # stats for different models
        for model_name in model_list:

            if model_name == 'CompenNet++_fast':
                train_option['max_iters'] = 1000
                train_option['batch_size'] = 24
                train_option['lr_drop_rate'] = 800
            elif model_name == 'CompenNet++_faster':
                train_option['max_iters'] = 500
                train_option['batch_size'] = 16
                train_option['lr_drop_rate'] = 400

            train_option['model_name'] = model_name.replace('/', '_')

            # stats for different loss functions
            for loss in loss_list:
                log_file = open(fullfile(log_dir, log_file_name), 'a')

                # set seed of rng for repeatability
                resetRNGseed(0)

                # create a WarpingNet model
                warping_net = CompenNetPlusplusModel.WarpingNet(with_refine='w/o_refine' not in model_name)

                # initialize WarpingNet with affine transformation (remember grid_sample is inverse warp, so src is the the desired warp
                src_pts = np.array([[-1, -1], [1, -1], [1, 1]]).astype(np.float32)
                dst_pts = np.array(mask_corners[0][0:3]).astype(np.float32)
                affine_mat = cv.getAffineTransform(src_pts, dst_pts)
                warping_net.set_affine(affine_mat.flatten())
                if torch.cuda.device_count() >= 1: warping_net = nn.DataParallel(warping_net, device_ids=device_ids).to(device)

                # create a CompenNet++ model from exisitng WarpingNet and CompenNet
                compen_net_pp = CompenNetPlusplusModel.CompenNetPlusplus(warping_net, compen_net)
                if torch.cuda.device_count() >= 1: compen_net_pp = nn.DataParallel(compen_net_pp, device_ids=device_ids).to(device)

                # train option for current configuration, i.e., data name and loss function
                train_option['data_name'] = data_name.replace('/', '_')
                train_option['loss'] = loss

                print('-------------------------------------- Training Options -----------------------------------')
                print('\n'.join('{}: {}'.format(k, v) for k, v in train_option.items()))
                print('------------------------------------ Start training {:s} ---------------------------'.format(model_name))
                compen_net_pp, valid_psnr, valid_rmse, valid_ssim = trainModel(compen_net_pp, train_data, valid_data, train_option)

                # save results to log file
                ret_str = '{:30s}{:<30}{:<20}{:<15}{:<15}{:<15}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}\n'
                cam_valid_resize = F.interpolate(cam_valid, prj_valid.shape[2:4])
                log_file.write(ret_str.format(data_name, model_name, loss, num_train, train_option['batch_size'], train_option['max_iters'],
                                              psnr(cam_valid_resize, prj_valid), rmse(cam_valid_resize, prj_valid), ssim(cam_valid_resize, prj_valid),
                                              valid_psnr, valid_rmse, valid_ssim))
                log_file.close()

                # [testing phase] create compensated testing images
                if save_compensation:
                    print('------------------------------------ Start testing {:s} ---------------------------'.format(model_name))
                    torch.cuda.empty_cache()

                    # desired test images are created such that they can fill the optimal displayable area (see paper for detail)
                    desire_test_path = fullfile(data_root, 'cam/desire/test')
                    assert os.path.isdir(desire_test_path), 'images and folder {:s} does not exist!'.format(desire_test_path)

                    # compensate and save images
                    desire_test = readImgsMT(desire_test_path).to(device)
                    cam_surf_test = cam_surf.expand_as(desire_test).to(device)
                    with torch.no_grad():
                        # simplify CompenNet++
                        compen_net_pp.module.simplify(cam_surf_test[0, ...].unsqueeze(0))

                        # compensate using CompenNet++
                        compen_net_pp.eval()
                        prj_cmp_test = compen_net_pp(desire_test, cam_surf_test).detach()  # compensated prj input image x^{*}
                    del desire_test, cam_surf_test

                    # create image save path
                    cmp_folder_name = '{}_{}_{}_{}_{}'.format(train_option['model_name'], loss, num_train, train_option['batch_size'],
                                                              train_option['max_iters'])
                    prj_cmp_path = fullfile(data_root, 'prj/cmp/test', cmp_folder_name)
                    if not os.path.exists(prj_cmp_path): os.makedirs(prj_cmp_path)

                    # save images
                    saveImgs(prj_cmp_test, prj_cmp_path)  # compensated testing images, i.e., to be projected to the surface
                    print('Compensation images saved to ' + prj_cmp_path)

                # clear cache
                del compen_net_pp, warping_net
                torch.cuda.empty_cache()
                print('-------------------------------------- Done! ---------------------------\n')
        del train_data
    del cam_valid, prj_valid

    print('All dataset done!')
