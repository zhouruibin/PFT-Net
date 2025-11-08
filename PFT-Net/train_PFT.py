#!/usr/bin/env python
import os
import random
import logging
import shutil

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from models.fewshot_PFT import FewShotSeg
from dataloaders.dataset_specifics import get_label_names
from dataloaders.datasets import TrainDataset as TrainDataset, TestDataset
from utils import *
from config import ex
import glob

# from focal_loss.focal_loss import FocalLoss


def find_latest_epoch_file(directory):
    # 查找目录下所有以epoch开头的.pth文件
    pattern = os.path.join(directory, '**', 'epoch*.pth')
    files = glob.glob(pattern, recursive=True)

    if not files:
        return None

    # 获取最新的文件
    latest_file = max(files, key=os.path.getmtime)

    return latest_file


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        # Set up source folder
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        # Set up logger -> log to .txt
        file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', f'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    # Deterministic setting for reproduciablity.
    if _config['seed'] is not None:
        random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        torch.cuda.manual_seed_all(_config['seed'])
        cudnn.deterministic = True

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'Create model...')
    model = FewShotSeg()
    model = model.cuda()
    model.train()

    _log.info(f'Set optimizer...')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    lr_milestones = [(ii + 1) * _config['max_iters_per_load'] for ii in
                     range(_config['n_steps'] // _config['max_iters_per_load'] - 1)]
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=_config['lr_step_gamma'])

    dataset = _config['dataset']
    now_cv = _config['eval_fold']
    _config['resume'] = find_latest_epoch_file(f'PFT/exps_on_{dataset}_setting1_PFT/train_{dataset}_cv{now_cv}')

    start_epoch = 0
    if _config['resume']:
        if os.path.isfile(_config['resume']):
            checkpoint = torch.load(_config['resume'], map_location=lambda storage, loc: storage.cuda())
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            _log.info("=> loaded checkpoint '{}' (epoch {})".format(_config['resume'], checkpoint['epoch']))
        else:
            _log.info("=> no checkpoint found at '{}'".format(_config['resume']))

    my_weight = torch.FloatTensor([0.1, 1.0]).cuda()
    criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)
    # criterion_focal = FocalLoss(gamma=0.7, weights=my_weight)

    _log.info(f'Load data...')
    data_config = {
        'data_dir': _config['path'][_config['dataset']]['data_dir'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'n_way': _config['n_way'],
        'n_query': _config['n_query'],
        'n_sv': _config['n_sv'],
        'max_iter': _config['max_iters_per_load'],
        'eval_fold': _config['eval_fold'],
        'min_size': _config['min_size'],
        'max_slices': _config['max_slices'],
        'test_label': _config['test_label'],
        'exclude_label': _config['exclude_label'],
        'use_gt': _config['use_gt'],
    }
    train_dataset = TrainDataset(data_config)
    train_loader = DataLoader(train_dataset,
                              batch_size=_config['batch_size'],
                              shuffle=True,
                              num_workers=_config['num_workers'],
                              pin_memory=True,
                              drop_last=True)

    if _config['use_validation']:
        data_config['supp_idx'] = _config['supp_idx']
        test_dataset = TestDataset(data_config)
        test_loader = DataLoader(test_dataset,
                                 batch_size=_config['batch_size'],
                                 shuffle=False,
                                 num_workers=_config['num_workers'],
                                 pin_memory=True,
                                 drop_last=True)

    n_sub_epochs = _config['n_steps'] // _config['max_iters_per_load']  # number of times for reloading
    log_loss = {'total_loss': 0, 'query_loss': 0, 'align_loss': 0, 'thresh_loss': 0, 'mse_loss': 0}

    i_iter = 0 + start_epoch * _config['max_iters_per_load']
    _log.info(f'Start training...')
    for sub_epoch in range(start_epoch, n_sub_epochs):
        _log.info(f'This is epoch "{sub_epoch}" of "{n_sub_epochs}" epochs.')

        for _, sample in enumerate(train_loader):
            # Prepare episode data.
            support_images = [[shot.float().cuda() for shot in way]
                              for way in sample['support_images']]
            support_fg_mask = [[shot.float().cuda() for shot in way]
                               for way in sample['support_fg_labels']]

            query_images = [query_image.float().cuda() for query_image in sample['query_images']]
            query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']], dim=0)

            query_pred = model(support_images, support_fg_mask, query_images, qry_mask=query_labels, train=True)
            # query_loss = criterion_focal(query_pred.permute(0, 2, 3, 1), query_labels)

            query_loss = criterion(torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps,
                                                         1 - torch.finfo(torch.float32).eps)), query_labels)

            loss = query_loss
            # Compute gradient and do SGD step.
            for param in model.parameters():
                param.grad = None

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()

            _run.log_scalar('total_loss', loss.item())
            _run.log_scalar('query_loss', query_loss)

            log_loss['total_loss'] += loss.item()
            log_loss['query_loss'] += query_loss

            if (i_iter + 1) % _config['print_interval'] == 0:
                total_loss = log_loss['total_loss'] / _config['print_interval']
                query_loss = log_loss['query_loss'] / _config['print_interval']

                log_loss['total_loss'] = 0
                log_loss['query_loss'] = 0

                _log.info(f'step {i_iter + 1}: total_loss: {total_loss}, query_loss: {query_loss}')

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save({'epoch': sub_epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'epoch{sub_epoch + 1}.pth'))
                # use validation
            if (i_iter + 1) % _config['val_every'] == 0 and _config['use_validation']:
                # Get unique labels (classes).
                labels = get_label_names(_config['dataset'])

                # Loop over classes.
                class_dice = {}
                class_iou = {}

                _log.info(f'Starting validation...')
                for label_val, label_name in labels.items():

                    # Skip BG class.
                    if label_name == 'BG':
                        continue
                    elif (not np.intersect1d([label_val], _config['test_label'])):
                        continue

                    # _log.info(f'Test Class: {label_name}')

                    # Get support sample + mask for current class.
                    val_support_sample = test_dataset.getSupport(label=label_val, all_slices=False,
                                                                 N=_config['n_part'])
                    test_dataset.label = label_val

                    # Test.
                    with torch.no_grad():
                        model.eval()

                        # Unpack support data.
                        val_support_image = [val_support_sample['image'][[i]].float().cuda() for i in
                                             range(val_support_sample['image'].shape[0])]  # n_shot x 3 x H x W
                        val_support_fg_mask = [val_support_sample['label'][[i]].float().cuda() for i in
                                               range(val_support_sample['image'].shape[0])]  # n_shot x H x W

                        # Loop through query volumes.
                        scores = Scores()
                        for i, val_sample in enumerate(test_loader):

                            # Unpack query data.
                            val_query_image = [val_sample['image'][i].float().cuda() for i in
                                               range(val_sample['image'].shape[0])]  # [C x 3 x H x W]
                            val_query_label = val_sample['label'].float().squeeze(dim=0).cuda()  # C x H x W

                            # Compute output.
                            # Match support slice and query sub-chunck.
                            val_query_pred = torch.zeros(val_query_label.shape[-3:])
                            C_q = val_sample['image'].shape[1]
                            idx_ = np.linspace(0, C_q, _config['n_part'] + 1).astype('int')
                            for sub_chunck in range(_config['n_part']):
                                support_image_s = [val_support_image[sub_chunck]]  # 1 x 3 x H x W
                                support_fg_mask_s = [val_support_fg_mask[sub_chunck]]  # 1 x H x W
                                query_image_s = val_query_image[0][
                                                idx_[sub_chunck]:idx_[sub_chunck + 1]]  # C' x 3 x H x W
                                query_label_s = val_query_label[
                                                idx_[sub_chunck]:idx_[sub_chunck + 1]]  # C' x 1 x H x W
                                query_pred_s = []
                                for i in range(query_image_s.shape[0]):
                                    _pred_s = model([support_image_s], [support_fg_mask_s], [query_image_s[[i]]],
                                                    train=False)  # C x 1 x H x W
                                    query_pred_s.append(_pred_s)
                                query_pred_s = torch.cat(query_pred_s, dim=0)
                                query_pred_s = query_pred_s.argmax(dim=1).cpu()  # C x H x W
                                val_query_pred[idx_[sub_chunck]:idx_[sub_chunck + 1]] = query_pred_s

                            # Record scores.
                            scores.record(val_query_pred, val_query_label.cpu())
                        # Log class-wise results
                        class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()
                        class_iou[label_name] = torch.tensor(scores.patient_iou).mean().item()

                class_iou['Mean'] = sum(class_iou.values()) / len(class_iou)
                class_dice['Mean'] = sum(class_dice.values()) / len(class_dice)
                _log.info(
                    f'====================================================model of epoch {sub_epoch + 1} test results===================================================')
                _log.info(f'Mean IoU: {class_iou}')
                _log.info(f'Mean Dice: {class_dice}')
                _log.info(
                    '=============================================================================================================================================')
                _log.info(f'End of validation.')

            i_iter += 1
    _log.info('End of training.')
    return 1