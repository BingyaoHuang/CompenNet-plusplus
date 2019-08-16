'''
CompenNet++ training functions
'''

from utils import *
import torch.nn.functional as F
import torch.optim as optim
import time
import visdom
import ImgProc

# for visualization
vis = visdom.Visdom()
assert vis.check_connection(), 'Visdom: No connection, start visdom first!'

l1_fun = nn.L1Loss()
l2_fun = nn.MSELoss()
ssim_fun = pytorch_ssim.SSIM()


# %% train network with data from multiple dataset
def trainModel(model, train_data, valid_data, train_option):
    device = train_option['device']

    # empty cuda cache before training
    if device.type == 'cuda': torch.cuda.empty_cache()

    # training data
    cam_surf_train = train_data['cam_surf']
    cam_train = train_data['cam_train']
    prj_train = train_data['prj_train']

    # list of parameters to be optimized
    params = filter(lambda param: param.requires_grad, model.parameters())  # only optimize parameters that require gradient

    # optimizer
    optimizer = optim.Adam(params, lr=train_option['lr'], weight_decay=train_option['l2_reg'])

    # learning rate drop scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_option['lr_drop_rate'], gamma=train_option['lr_drop_ratio'])

    # %% start train
    start_time = time.time()

    # get model name
    if not 'model_name' in train_option: train_option['model_name'] = model.name if hasattr(model, 'name') else model.module.name

    # initialize visdom data visualization figure
    if 'plot_on' not in train_option: train_option['plot_on'] = True

    if train_option['plot_on']:
        # title string of current training option
        title = optionToString(train_option)

        # intialize visdom figures
        vis_train_fig = None
        vis_valid_fig = None
        vis_curve_fig = vis.line(X=np.array([0]), Y=np.array([0]),
                                 opts=dict(title=title, font=dict(size=20), layoutopts=dict(
                                     plotly=dict(xaxis={'title': 'Iteration'}, yaxis={'title': 'Metrics', 'hoverformat': '.4f'})),
                                           width=1300, height=500, markers=True, markersize=3),
                                 name='origin')

    # main loop
    iters = 0

    while iters < train_option['max_iters']:
        # randomly sample training batch and send to GPU
        idx = random.sample(range(train_option['num_train']), train_option['batch_size'])
        cam_surf_train_batch = cam_surf_train[idx, :, :, :].to(device) if cam_surf_train.device.type != 'cuda' else cam_surf_train[idx, :, :, :]
        cam_train_batch = cam_train[idx, :, :, :].to(device) if cam_train.device.type != 'cuda' else cam_train[idx, :, :, :]
        prj_train_batch = prj_train[idx, :, :, :].to(device) if prj_train.device.type != 'cuda' else prj_train[idx, :, :, :]

        # predict and compute loss
        model.train()  # explicitly set to train mode in case batchNormalization and dropout are used
        prj_train_pred = predict(model, dict(cam=cam_train_batch, cam_surf=cam_surf_train_batch))
        train_loss_batch, train_l2_loss_batch = computeLoss(prj_train_pred, prj_train_batch, train_option['loss'])
        train_rmse_batch = math.sqrt(train_l2_loss_batch.item() * 3)  # 3 channels, rgb

        # backpropagation and update params
        optimizer.zero_grad()
        train_loss_batch.backward()
        optimizer.step()

        # record running time
        time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))

        # plot train
        if train_option['plot_on']:
            if iters % train_option['train_plot_rate'] == 0 or iters == train_option['max_iters'] - 1:
                vis_train_fig = plotMontage(dict(cam=cam_train_batch.detach(), pred=prj_train_pred.detach(), prj=prj_train_batch.detach()),
                                            win=vis_train_fig, title='[Train]' + title)
                appendDataPoint(iters, train_loss_batch.item(), vis_curve_fig, 'train_loss')
                appendDataPoint(iters, train_rmse_batch, vis_curve_fig, 'train_rmse')

        # validation
        valid_psnr, valid_rmse, valid_ssim = 0., 0., 0.
        if valid_data is not None and (iters % train_option['valid_rate'] == 0 or iters == train_option['max_iters'] - 1):
            valid_psnr, valid_rmse, valid_ssim, prj_valid_pred = evaluate(model, valid_data)

            # plot validation
            if train_option['plot_on']:
                vis_valid_fig = plotMontage(dict(cam=valid_data['cam_valid'], pred=prj_valid_pred, prj=valid_data['prj_valid']),
                                            win=vis_valid_fig, title='[Valid]' + title)
                appendDataPoint(iters, valid_rmse, vis_curve_fig, 'valid_rmse')
                appendDataPoint(iters, valid_ssim, vis_curve_fig, 'valid_ssim')

        # print to console
        print('Iter:{:5d} | Time: {} | Train Loss: {:.4f} | Train RMSE: {:.4f} | Valid PSNR: {:7s}  | Valid RMSE: {:6s}  '
              '| Valid SSIM: {:6s}  | Learn Rate: {:.5f} |'.format(iters, time_lapse, train_loss_batch.item(), train_rmse_batch,
                                                                   '{:>2.4f}'.format(valid_psnr) if valid_psnr else '',
                                                                   '{:.4f}'.format(valid_rmse) if valid_rmse else '',
                                                                   '{:.4f}'.format(valid_ssim) if valid_ssim else '',
                                                                   optimizer.param_groups[0]['lr']))

        lr_scheduler.step()  # update learning rate according to schedule
        iters += 1

    # Done training and save the last epoch model
    checkpoint_dir = '../../checkpoint'
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    checkpoint_file_name = fullfile(checkpoint_dir, title + '.pth')
    torch.save(model.state_dict(), checkpoint_file_name)
    print('Checkpoint saved to {}\n'.format(checkpoint_file_name))

    return model, valid_psnr, valid_rmse, valid_ssim


# %% load training and validation data
def loadData(dataset_root, data_name, input_size):
    # data paths
    data_root = fullfile(dataset_root, data_name)
    cam_ref_path = fullfile(data_root, 'cam/raw/ref')
    cam_train_path = fullfile(data_root, 'cam/raw/train')
    prj_train_path = fullfile(dataset_root, 'train')
    cam_valid_path = fullfile(data_root, 'cam/raw/test')
    prj_valid_path = fullfile(dataset_root, 'test')
    print("Loading data from '{}'".format(data_root))

    # training data
    cam_surf = readImgsMT(cam_ref_path, index=[125])  # ref/img_0126.png is cam-captured surface image i.e., s when img_gray.png i.e., x0 projected
    cam_train = readImgsMT(cam_train_path)
    prj_train = readImgsMT(prj_train_path)

    # validation data
    cam_valid = readImgsMT(cam_valid_path)
    prj_valid = readImgsMT(prj_valid_path)

    # find projector FOV mask
    im_diff = readImgsMT(cam_ref_path, index=[124], size=input_size) - readImgsMT(cam_ref_path, index=[0], size=input_size)
    im_diff = im_diff.numpy().transpose((2, 3, 1, 0))
    prj_fov_mask = torch.zeros(cam_surf.shape)

    # threshold im_diff with Otsu's method
    mask_corners = [None] * im_diff.shape[-1]
    for i in range(im_diff.shape[-1]):
        im_mask, mask_corners[i] = ImgProc.thresh(im_diff[:, :, :, i])
        prj_fov_mask[i, :, :, :] = repeat_np(torch.Tensor(np.uint8(im_mask)).unsqueeze(0), 3, 0)

    prj_fov_mask = prj_fov_mask.byte()

    # mask out background areas that are out of projector's FOV
    cam_surf[~prj_fov_mask] = 0

    cam_train = torch.where(prj_fov_mask, cam_train, torch.tensor([0.]))
    cam_valid = torch.where(prj_fov_mask, cam_valid, torch.tensor([0.]))

    return cam_surf, cam_train, cam_valid, prj_train, prj_valid, mask_corners

# initialize CompenNet to |x-s| without actual projections
def initCompenNet(compen_net, dataset_root, device):
    ckpt_file = '../../checkpoint/init_CompenNet_l1+ssim_500_48_500_0.001_0.2_800_0.0001.pth'

    if os.path.exists(ckpt_file):
        # load weights initialized CompenNet from saved state dict
        compen_net.load_state_dict(torch.load(ckpt_file))

        print('CompenNet state dict found! Loading...')
    else:
        # initialize the model if checkpoint file does not exist
        print('CompenNet state dict not found! Initializing...')
        cam_surf_init_path = fullfile(dataset_root, 'init')
        prj_train_path = fullfile(dataset_root, 'train')

        # load data
        cam_surf = readImgsMT(cam_surf_init_path)
        prj_train = readImgsMT(prj_train_path)
        init_data = dict(cam_surf=cam_surf.expand_as(prj_train),
                         cam_train=torch.abs(prj_train - 0.3 * cam_surf.expand_as(prj_train)),
                         prj_train=prj_train)

        # then initialize compenNet to |x-s|
        init_option = {'data_name': 'init', 'num_dataset': 1, 'device': device, 'max_epochs': 2000, 'max_iters': 500, 'batch_size': 48, 'lr': 1e-3,
                       'lr_drop_ratio': 0.2, 'lr_drop_rate': 800, 'loss': 'l1+ssim', 'l2_reg': 1e-4, 'plot_on': True, 'train_plot_rate': 50,
                       'valid_rate': 200, 'num_train': 500}

        compen_net, _, _, _ = trainModel(compen_net, init_data, None, init_option)

    return compen_net


# compute loss between prediction and ground truth
def computeLoss(prj_pred, prj_train, loss_option):
    # l1
    l1_loss = 0
    if 'l1' in loss_option:
        l1_loss = l1_fun(prj_pred, prj_train)

    # l2
    l2_loss = l2_fun(prj_pred, prj_train)

    # ssim
    ssim_loss = 0
    if 'ssim' in loss_option:
        ssim_loss = 1 * (1 - ssim_fun(prj_pred, prj_train))

    train_loss = 0
    # linear combination of losses
    if loss_option == 'l1':
        train_loss = l1_loss
    elif loss_option == 'l2':
        train_loss = l2_loss
    elif loss_option == 'l1+l2':
        train_loss = l1_loss + l2_loss
    elif loss_option == 'ssim':
        train_loss = ssim_loss
    elif loss_option == 'l1+ssim':
        train_loss = l1_loss + ssim_loss
    elif loss_option == 'l2+ssim':
        train_loss = l2_loss + ssim_loss
    elif loss_option == 'l1+l2+ssim':
        train_loss = l1_loss + l2_loss + ssim_loss
    else:
        print('Unsupported loss')

    return train_loss, l2_loss


# append a data point to the curve in win
def appendDataPoint(x, y, win, name, env=None):
    vis.line(
        X=np.array([x]),
        Y=np.array([y]),
        env=env,
        win=win,
        update='append',
        name=name,
        opts=dict(markers=True, markersize=3)
    )


# plot sample predicted images using visdom
def plotMontage(images, win=None, title=None, env=None):
    cam_im = images['cam']  # Camera catpured uncompensated images
    prj_pred = images['pred']  # CompenNet predicted projector input images
    prj_im = images['prj']  # Ground truth of projector input images

    if cam_im.device.type == 'cpu' or prj_pred.device.type == 'cpu' or prj_im.device.type == 'cpu':
        cam_im = cam_im.cpu()
        prj_pred = prj_pred.cpu()
        prj_im = prj_im.cpu()

    # compute montage grid size
    # step = 1
    if cam_im.shape[0] > 5:
        grid_w = 5
        # step = round(cam_im.shape[0] / grid_w)
        # grid_w = len(range(0, cam_im.shape[0], step))
        idx = random.sample(range(0, cam_im.shape[0]), grid_w)
    else:
        grid_w = cam_im.shape[0]

    # resize if the image sizes are not the same
    # if cam_im.shape != prj_im.shape:
    #     cam_im_resize = F.interpolate(cam_im[::step, :, :, :], (prj_im.shape[2:4]))
    # else:
    #     cam_im_resize = cam_im[::step, :, :, :]

    # resize to (256, 256) for better display
    tile_size = (256, 256)
    if cam_im.shape[2] != tile_size[0] or cam_im.shape[3] != tile_size[1]:
        cam_im_resize = F.interpolate(cam_im[idx, :, :, :], tile_size)
    else:
        cam_im_resize = cam_im[idx, :, :, :]
    if prj_im.shape[2] != tile_size[0] or prj_im.shape[3] != tile_size[1]:
        prj_im_resize = F.interpolate(prj_im[idx, :, :, :], tile_size)
    else:
        prj_im_resize = prj_im[idx, :, :, :]
    if prj_pred.shape[2] != tile_size[0] or prj_pred.shape[3] != tile_size[1]:
        prj_pred_resize = F.interpolate(prj_pred[idx, :, :, :], tile_size)
    else:
        prj_pred_resize = prj_pred[idx, :, :, :]

    # % view results
    im_concat = torch.cat((cam_im_resize, prj_pred_resize, prj_im_resize), 0)
    im_montage = montage(im_concat, multichannel=True, padding_width=10, fill=[1, 1, 1], grid_shape=(3, grid_w))

    # title
    plot_opts = dict(title=title, caption=title, font=dict(size=18))

    # plot montage to existing win, otherwise create a new win
    win = vis.image(im_montage.transpose(2, 0, 1), win=win, opts=plot_opts, env=env)
    return win


# predict projector input images given input data (do not use with torch.no_grad() within this function)
def predict(model, data):
    if 'cam_surf' in data and data['cam_surf'] is not None:
        prj_pred = model(data['cam'], data['cam_surf'])
    else:
        prj_pred = model(data['cam'])

    if type(prj_pred) == tuple and len(prj_pred) > 1: prj_pred = prj_pred[0]
    return prj_pred


# evaluate model on validation dataset
def evaluate(model, valid_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cam_surf = valid_data['cam_surf']
    cam_valid = valid_data['cam_valid']
    prj_valid = valid_data['prj_valid']

    with torch.no_grad():
        model.eval()  # explicitly set to eval mode

        # if you have limited GPU memory, we need to predict in batch mode
        if cam_surf.device.type != device.type:
            last_loc = 0
            valid_mse, valid_ssim = 0., 0.

            prj_valid_pred = torch.zeros(prj_valid.shape)
            num_valid = cam_valid.shape[0]
            batch_size = 50 if num_valid > 50 else num_valid  # default number of test images per dataset

            for i in range(0, num_valid // batch_size):
                idx = range(last_loc, last_loc + batch_size)
                cam_surf_batch = cam_surf[idx, :, :, :].to(device) if cam_surf.device.type != 'cuda' else cam_surf[idx, :, :, :]
                cam_valid_batch = cam_valid[idx, :, :, :].to(device) if cam_valid.device.type != 'cuda' else cam_valid[idx, :, :, :]
                prj_valid_batch = prj_valid[idx, :, :, :].to(device) if prj_valid.device.type != 'cuda' else prj_valid[idx, :, :, :]

                # predict batch
                prj_valid_pred_batch = predict(model, dict(cam=cam_valid_batch, cam_surf=cam_surf_batch)).detach()
                if type(prj_valid_pred_batch) == tuple and len(prj_valid_pred_batch) > 1: prj_valid_pred_batch = prj_valid_pred_batch[0]
                prj_valid_pred[last_loc:last_loc + batch_size, :, :, :] = prj_valid_pred_batch.cpu()

                # compute loss
                valid_mse += l2_fun(prj_valid_pred_batch, prj_valid_batch).item() * batch_size
                valid_ssim += ssim(prj_valid_pred_batch, prj_valid_batch) * batch_size

                last_loc += batch_size
            # average
            valid_mse /= num_valid
            valid_ssim /= num_valid

            valid_rmse = math.sqrt(valid_mse * 3)  # 3 channel image
            valid_psnr = 10 * math.log10(1 / valid_mse)
        else:
            # if all data can be loaded to GPU memory
            prj_valid_pred = predict(model, dict(cam=cam_valid, cam_surf=cam_surf)).detach()
            valid_mse = l2_fun(prj_valid_pred, prj_valid).item()
            valid_rmse = math.sqrt(valid_mse * 3)  # 3 channel image
            valid_psnr = 10 * math.log10(1 / valid_mse)
            valid_ssim = ssim_fun(prj_valid_pred, prj_valid).item()

    return valid_psnr, valid_rmse, valid_ssim, prj_valid_pred


# generate training title string
def optionToString(train_option):
    return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(train_option['data_name'], train_option['model_name'], train_option['loss'],
                                                  train_option['num_train'], train_option['batch_size'], train_option['max_iters'],
                                                  train_option['lr'], train_option['lr_drop_ratio'], train_option['lr_drop_rate'],
                                                  train_option['l2_reg'])
