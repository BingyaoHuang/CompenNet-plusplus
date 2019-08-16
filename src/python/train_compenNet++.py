'''
Train CompenNet++
'''
from trainNetwork import *
import CompenNetPlusplusModel
import ImgProc
from time import localtime, strftime

# % main script
# set device
lab3 = 1
if lab3:
    print("Lab3")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2, 3'
    device_ids = [0, 1, 2]
else:
    # lab 1
    print("Lab1")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 3'
    device_ids = [0, 1]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() >= 1:
    print("Train with", torch.cuda.device_count(), "GPUs!")
else:
    print("Train with CPUs!")

# repeatability
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% all train options

dataset_root = fullfile(os.getcwd(), '../../Dataset')

data_list = [
    'canonCamlink/dark/pos1/cloud_np',
    'canonCamlink/dark/pos1/lavender_np',
    'canonCamlink/dark/pos2/cubes_np',
    'canonCamlink/dark/pos3/bars_spec_np',
    'canonCamlink/dark/pos3/bubbles_np',
    'canonCamlink/dark/pos3/curves_np',
    'canonCamlink/dark/pos3/lavender_spec_np',
    'canonCamlink/dark/pos5/curves_np',
    'canonCamlink/dark/pos5/lavender_np',
    'canonCamlink/dark/pos5/stripes_np',
    'canonCamlink/dark/pos6/cubes_np',
    'canonCamlink/dark/pos6/curves_np',
    'canonCamlink/dark/pos6/lavender_np',
    'canonCamlink/dark/pos6/pillow_np',
    'canonCamlink/dark/pos6/stripes_np',
    'canonCamlink/light/pos4/bubbles_np',
    'canonCamlink/light/pos4/cloud_np',
    'canonCamlink/light/pos4/curves_np',
    'canonCamlink/light/pos4/squares_np',
    'canonCamlink/light/pos4/water_np',
]



# print(data_list)

# num_train_list = [125, 250, 500]
num_train_list = [48]

train_warp_option = {'data_name': '', 'device': device, 'max_epochs': 2000, 'max_iters': 100,
                     'batch_size': 48,
                     'lr': 1e-3,
                     'lr_drop_ratio': 0.2,
                     'lr_drop_rate': 500, 'loss': '', 'l2_reg': 1e-4, 'valid_rate': 100, 'train_plot_rate': 50, 'plot_on': False}

init_compen_option = {'data_name': '', 'device': device, 'max_epochs': 2000, 'max_iters': 400,
                      'batch_size': 48,
                      'lr': 1e-3,
                      'lr_drop_ratio': 0.2,
                      'lr_drop_rate': 200, 'loss': '', 'l2_reg': 1e-4, 'valid_rate': 100, 'train_plot_rate': 50}

# normal compenNet++ training option
train_compen_option = {'data_name': '', 'device': device, 'max_epochs': 2000, 'max_iters': 1500,
                       'batch_size': 48,
                       'lr': 1e-3,
                       'lr_drop_ratio': 0.2,
                       'lr_drop_rate': 1000, 'loss': '', 'l2_reg': 1e-4, 'valid_rate': 1500, 'train_plot_rate': 50}

# for iccv19 rebuttal: CompenNet++ fast
# train_compen_option = {'data_name': '', 'device': device, 'max_epochs': 2000, 'max_iters': 1000,
#                        'batch_size': 24,
#                        'lr': 1e-3,
#                        'lr_drop_ratio': 0.2,
#                        'lr_drop_rate': 800, 'loss': '', 'l2_reg': 1e-4, 'valid_rate': 500, 'train_plot_rate': 50}

# for iccv19 rebuttal: CompenNet++ faster
# train_compen_option = {'data_name': '', 'device': device, 'max_epochs': 2000, 'max_iters': 500,
#                        'batch_size': 16,
#                        'lr': 1e-3,
#                        'lr_drop_ratio': 0.2,
#                        'lr_drop_rate': 400, 'loss': '', 'l2_reg': 1e-4, 'valid_rate': 500, 'train_plot_rate': 50}

# loss_list = ['l1', 'l2', 'ssim', 'l1+ssim']
loss_list = ['l1+ssim']

# model options
# model_list = ['warpRefineNet2','warpNet']
# model_list = ['warpNet']
model_list = ['warpRefineNet2']

compenNet_only = False

# net_opts = {'use_stn': True, 'use_affine': True, 'use_tps': True, 'grid_shape': (4, 4), 'out_size': (256, 256)}
net_opts = {'use_stn': False, 'use_affine': True, 'use_tps': True, 'grid_shape': (6, 6), 'out_size': (256, 256)}

# a file to save results
log_file_name = strftime("%Y-%m-%d_%H_%M_%S", localtime()) + '.txt'
log_file = open(log_file_name, "w")
log_file.write(
    '{:30s} {}   {}   {}   {}   {}   {}   {}   {}   {}   {}    {}\n'.format('data_name', 'model_name', 'num_train', 'batch_size', 'loss_function',
                                                                            'cam_psnr', 'cam_rmse',
                                                                            'cam_ssim', 'predict_psnr', 'predict_rmse', 'predict_ssim', 'max_iters'))
log_file.close()

input_size = None
# input_size = (480, 640)


sl_warp = True
use_mask = True
pretrain_warp = False
cap_frame = True

# for fast debug
DEBUG = False
# DEBUG = True
if DEBUG:
    print('--------------Debug mode!--------------')
    idx = range(10)
    train_warp_option['batch_size'] = 3
    init_compen_option['batch_size'] = 3
    train_compen_option['batch_size'] = 3

    train_warp_option['num_train'] = len(idx)
    init_compen_option['num_train'] = len(idx)
    train_compen_option['num_train'] = len(idx)

    # for param in warp_net.parameters():
    #     param.require_grad = True
    # tst = warp_net(cam_valid, cam_surf_valid.to(device))
    # print(ssim(tst, prj_valid))
    # print(rmse(tst, prj_valid))
else:
    idx = None

# %% main loop
for data_name in data_list:
    # data paths
    data_root = fullfile(dataset_root, data_name)
    cam_surf_path = fullfile(data_root, 'cam/warp/ref') if compenNet_only else fullfile(data_root, 'cam/raw/ref')
    cam_train_path = fullfile(data_root, 'cam/warp/train') if compenNet_only else fullfile(data_root, 'cam/raw/train')
    prj_train_path = fullfile(dataset_root, 'train')
    cam_valid_path = fullfile(data_root, 'cam/warp/test') if compenNet_only else fullfile(data_root, 'cam/raw/test')
    prj_valid_path = fullfile(dataset_root, 'test')


    if sl_warp:
        cam_surf_path = cam_surf_path.replace('warp', 'warpSL')
        cam_train_path = cam_train_path.replace('warp', 'warpSL')
        cam_valid_path = cam_valid_path.replace('warp', 'warpSL')
    print('loading data from {}'.format(data_root))

    # load data
    # cam_surf = readImgsMT(cam_surf_path, index=[62], size=input_size)
    cam_surf = readImgsMT(cam_surf_path, index=[125], size=input_size)
    cam_train = readImgsMT(cam_train_path, index=idx, size=input_size)
    prj_train = readImgsMT(prj_train_path, index=idx)
    cam_valid = readImgsMT(cam_valid_path, index=idx, size=input_size)
    prj_valid = readImgsMT(prj_valid_path, index=idx)

    if use_mask and not compenNet_only:
        # find projector fov mask
        im_diff = readSelectedImgsMultiDataset(cam_surf_path, index=[124], size=input_size) - readSelectedImgsMultiDataset(cam_surf_path, index=[0],
                                                                                                                           size=input_size)
        im_diff = im_diff.numpy().transpose((2, 3, 1, 0))

        prj_fov_mask = torch.zeros(cam_surf.shape)

        # threshold im_diff with Otsu's method
        mask_corners = [None] * im_diff.shape[-1]
        for i in range(im_diff.shape[-1]):
            im_mask, mask_corners[i] = ImgProc.thresh(im_diff[:, :, :, i])
            prj_fov_mask[i, :, :, :] = repeat_np(torch.Tensor(np.uint8(im_mask)).unsqueeze(0), 3, 0)

        prj_fov_mask = prj_fov_mask.byte()

        # mask out background areas that are out of projector's fov
        cam_surf[~prj_fov_mask] = 0
        # cam_train[~prj_fov_mask.expand_as(cam_train)] = 0  # very slow
        # cam_valid[~prj_fov_mask.expand_as(cam_valid)] = 0

        cam_train = torch.where(prj_fov_mask, cam_train, torch.tensor([0.]))
        cam_valid = torch.where(prj_fov_mask, cam_valid, torch.tensor([0.]))

    cam_surf_train = cam_surf.expand_as(cam_train)
    cam_surf_valid = cam_surf.expand_as(cam_valid)

    # convert valid data to CUDA tensor if you have sufficient GPU memory (huge speedup)
    cam_valid.to(device)
    prj_valid.to(device)

    # for validation. 200 images
    valid_data = dict(cam_surf=cam_surf_valid, cam_valid=cam_valid, prj_valid=prj_valid)

    # load initialization surface data for compenNet init
    if 'compen_net' not in locals():

        if 0:
            cam_surf_init_path = fullfile(dataset_root, 'init')
            #             cam_surf_init = readSelectedImgsMultiDataset(cam_surf_init_path, size=input_size)
            #
            #             # then initialize compenNet to |x-s|
            #             init_compen_option['data_name'] = 'init'
            #             init_compen_option['loss'] = 'l1+ssim'
            #             init_compen_option['num_train'] = 500
            #             print('-----------------------------------------Initializing compenNet---------------------------\n' + str(init_compen_option))
            #
            #             # used to initialize compenNet to |x-s|
            #             init_data = dict(cam_surf=cam_surf_init.expand_as(prj_train),
            #                              cam_train=torch.abs(prj_train - 0.3 * cam_surf_init.expand_as(prj_train)),
            #                              prj_train=prj_train)
            #             # init_data = dict(cam_surf=prj_train[:num_train, :, :, :], cam_train=prj_train[:num_train, :, :, :], prj_train=prj_train[:num_train, :, :, :])
            #             # init_data = dict(cam_surf=cam_surf_init.expand_as(prj_train[:num_train, :, :, :]),
            #             #                  cam_train=torch.abs(prj_train[:num_train, :, :, :]-0.3*cam_surf_init.expand_as(prj_train[:num_train, :, :, :])),
            #             #                  prj_train=prj_train[:num_train, :, :, :])
        resetRNGseed(0)
        compen_net = Models.compenNet()
        if torch.cuda.device_count() >= 1:
            compen_net = nn.DataParallel(compen_net, device_ids=device_ids)
        compen_net.to(device)

        # load weight initialized compenNet
        compen_net.load_state_dict(torch.load('../../checkpoint/init_compenNet_l1+ssim_500_48_1000_0.001_0.2_800_0.0001.pth'))
        # compen_net, valid_psnr, valid_rmse, valid_ssim = trainNetworkMultiData(compen_net, init_data, None, init_compen_option)

    # for different #training imgs
    for num_train in num_train_list:
        if not DEBUG:
            train_warp_option['num_train'] = num_train
            init_compen_option['num_train'] = num_train
            train_compen_option['num_train'] = num_train

        # select a subset to train
        train_data = dict(cam_surf=cam_surf_train[:num_train, :, :, :], cam_train=cam_train[:num_train, :, :, :],
                          prj_train=prj_train[:num_train, :, :, :])

        for model_name in model_list:
            for loss in loss_list:
                print('Current training setting: | [model_name]: {} | [data_name]: {} | [num_train]: {}  | [loss_function]: {} |'.format(model_name,
                                                                                                                                         data_name,
                                                                                                                                         num_train,
                                                                                                                                         loss))
                log_file = open(log_file_name, "a")

                # set seed of rng for repeatability
                resetRNGseed(0)

                # define warpNet
                if not compenNet_only:
                    warp_net = getattr(Models, model_name)(grid_shape=net_opts['grid_shape'],
                                                           out_size=net_opts['out_size'],
                                                           use_stn=net_opts['use_stn'],
                                                           use_affine=net_opts['use_affine'],
                                                           use_tps=net_opts['use_tps'])

                    # initialize warp net with affine transformation (remember grid_sample is inverse warp, so src is the the desired warp
                    src_pts = np.array([[-1, -1], [1, -1], [1, 1]]).astype(np.float32)
                    dst_pts = np.array(mask_corners[0][0:3]).astype(np.float32)
                    affine_mat = cv.getAffineTransform(src_pts, dst_pts)
                    warp_net.set_affine(affine_mat.flatten())

                    if torch.cuda.device_count() >= 1:
                        warp_net = nn.DataParallel(warp_net, device_ids=device_ids)
                    warp_net.to(device)

                    if pretrain_warp:
                        # % train warp_net
                        train_warp_option['data_name'] = data_name.replace('/', '_')
                        train_warp_option['loss'] = loss
                        train_warp_option['num_train'] = 5
                        train_warp_option['batch_size'] = 5
                        # train_warp_option['plot_on'] = False

                        # train_warp_data = dict(cam_surf=prj_fov_mask.expand_as(train_data['cam_surf']).float(), cam_train=prj_fov_mask.expand_as(train_data['cam_train']).float(),
                        #                   prj_train=torch.ones(train_data['prj_train'].shape))
                        train_warp_size = torch.Size((train_warp_option['num_train'], 3)) + train_data['cam_surf'].shape[2:4]
                        tmp_cam_surf = prj_fov_mask.expand(train_warp_size).float()
                        train_warp_data = dict(cam_surf=tmp_cam_surf, cam_train=tmp_cam_surf,
                                               prj_train=torch.ones(train_warp_size[0:2] + train_data['prj_train'].shape[2:4]))

                        print(train_warp_option)
                        warp_net, valid_psnr, valid_rmse, valid_ssim = trainModel(warp_net, train_warp_data, valid_data, train_warp_option)

                # then train compenWarpNet
                train_compen_option['data_name'] = data_name.replace('/', '_')
                train_compen_option['loss'] = loss
                print('-----------------------------------------Train compenWarpNet---------------------------\n' + str(train_compen_option))

                if compenNet_only:
                    compen_net_train = Models.compenNet()
                    # compen_net_train.module = copy.deepcopy(compen_net.module)  # do not copy weight initialized compenNet
                    if torch.cuda.device_count() >= 1:
                        compen_net_train = nn.DataParallel(compen_net_train, device_ids=device_ids)
                    compen_net_train.to(device)
                    compen_net_train, valid_psnr, valid_rmse, valid_ssim = trainModel(compen_net_train, train_data, valid_data,
                                                                                      train_compen_option)

                    uncmp_valid_psnr = psnr(cam_valid, prj_valid)
                    uncmp_valid_rmse = rmse(cam_valid, prj_valid)
                    uncmp_valid_ssim = ssim(cam_valid, prj_valid)

                    # save results
                    log_file.write(
                        '{:30s}  {:40s}  {:4d}  {:4d}   {:10s}   {:>2.4f}   {:.4f}   {:.4f}   {:>2.4f}   {:.4f}   {:.4f}   {:4d} \n'.format(
                            data_name, compen_net_train.module.name, num_train, train_compen_option['batch_size'], loss, uncmp_valid_psnr,
                            uncmp_valid_rmse, uncmp_valid_ssim, valid_psnr,
                            valid_rmse, valid_ssim, train_compen_option['max_iters']))
                else:
                    compenwarp_net = Models.compenWarpNet(warp_net, compen_net, fix_warp=False)
                    if torch.cuda.device_count() >= 1:
                        compenwarp_net = nn.DataParallel(compenwarp_net, device_ids=device_ids)
                    compenwarp_net.to(device)
                    compenwarp_net.module.name = warp_net.module.name + '_init'
                    compenwarp_net, valid_psnr, valid_rmse, valid_ssim = trainModel(compenwarp_net, train_data, valid_data,
                                                                                    train_compen_option)

                    # fine-tuned warp only
                    warp_uncmp_valid_psnr, warp_uncmp_valid_rmse, warp_uncmp_valid_ssim, prj_warp_valid_pred = evaluate(compenwarp_net.module.warpNet,
                                                                                                                        valid_data)

                    # save results
                    log_file.write(
                        '{:30s}  {:40s}  {:4d}  {:4d}   {:10s}   {:>2.4f}   {:.4f}   {:.4f}   {:>2.4f}   {:.4f}   {:.4f}   {:4d} \n'.format(
                            data_name, compenwarp_net.module.name, num_train, train_compen_option['batch_size'], loss, warp_uncmp_valid_psnr,
                            warp_uncmp_valid_rmse, warp_uncmp_valid_ssim, valid_psnr,
                            valid_rmse, valid_ssim, train_compen_option['max_iters']))
                log_file.close()

                #% create compensated testing images
                if 0:
                    torch.cuda.empty_cache()
                    if compenNet_only:
                        net = compen_net_train
                        cam_test_path = fullfile(data_root, 'cam/warpSL/desire/test')
                        cam_frames_path = fullfile(data_root, 'cam/warpSL/desire/frames')
                    else:
                        net = compenwarp_net
                        cam_test_path = fullfile(data_root, 'cam/desire/test')
                        cam_frames_path = fullfile(data_root, 'cam/desire/frames')

                    model_full_name = "{}_{}_{}_{}_{}".format(loss, num_train, train_warp_option['max_iters'],
                                                              train_warp_option['batch_size'], net.module.name)

                    # some datasets don't have desire folder
                    if os.path.isdir(cam_test_path):
                        with torch.no_grad():
                            # load data for AR
                            cam_test = readImgsMT(cam_test_path)
                            cam_test = cam_test.to(device)
                            cam_surf_test = cam_surf.expand_as(cam_test).to(device)
                            prj_cmp_test = net(cam_test, cam_surf_test).detach()  # for geometric warping only
                            del cam_test, cam_surf_test

                            # create image save path
                            prj_cmp_test_path = fullfile(data_root, 'prj/cmp/test', model_full_name)
                            if not os.path.exists(prj_cmp_test_path):
                                os.makedirs(prj_cmp_test_path)
                            # save images
                            saveImgs(prj_cmp_test, prj_cmp_test_path)

                            # frames
                            if cap_frame:
                                cam_frames = readImgsMT(cam_frames_path)
                                cam_frames = cam_frames.to(device)
                                cam_surf_frames = cam_surf.expand_as(cam_frames).to(device)
                                prj_cmp_frames = net(cam_frames, cam_surf_frames).detach()  # for geometric warping only
                                del cam_frames, cam_surf_frames

                                prj_cmp_frames_path = fullfile(data_root, 'prj/cmp/frames', model_full_name)
                                if not os.path.exists(prj_cmp_frames_path):
                                    os.makedirs(prj_cmp_frames_path)
                                saveImgs(prj_cmp_frames, prj_cmp_frames_path)

                # clear cache
                # del warp_net, compenwarp_net
                torch.cuda.empty_cache()
        del train_data
    del cam_valid, prj_valid

    print("Done all training test!")
