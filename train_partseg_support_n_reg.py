"""
Author: Benny
Date: Nov 2019
"""
# python train_semseg_support.py --model pointnet2_sem_seg --log_dir pointnet2_seg_msg_support --normal --batch_size 2
import argparse
import os
from data_utils.SupportDataLoader import RegSupportDataset, my_collate_fn
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from tensorboardX import SummaryWriter
writer = SummaryWriter()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['other', '1']  # , '2', '3'
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

seg_classes = {'toothModel': [0, 1]}   # , 2, 3
seg_label_to_cat = {}   # {0:toothModel, 1:toothModel, 2:toothModel, 3:toothModel}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

 
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(), ]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=300, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.005, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=40000, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.8, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('part_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = r"/data/support/downsample0325/"  # r"/home/heygears/jinhai_zhou/data/pcd_with_label"
    NUM_CLASSES = 2
    # f_cols = 3
    # if args.normal:
    #     f_cols = 6
    f_cols = 12
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    train_val_ratio = 0.9

    # print("start loading training data ...")
    # TRAIN_DATASET = SemSegDataset(split='train', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    # print("start loading test data ...")
    # TEST_DATASET = SemSegDataset(split='test', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    #
    # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
    #                                               pin_memory=True, drop_last=True,
    #                                               worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    # testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
    #                                              pin_memory=True, drop_last=True)
    # TRAIN_DATASET = SemSegSupportDataset(root=root, npoints=args.npoint, split='train_val', normal_channel=args.normal, n_class=NUM_CLASSES, f_cols=f_cols)
    # TEST_DATASET = SemSegSupportDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal, shuffle=False, n_class=NUM_CLASSES, f_cols=f_cols)
    
    # dataset = SemSegSupportDataset(root=root, npoints=args.npoint, split='train_val', normal_channel=args.normal, n_class=NUM_CLASSES, f_cols=f_cols)
    # n_train = int(train_val_ratio * len(dataset))
    # n_val = len(dataset) - n_train
    # print("data number: {}, train: {}, val: {}".format(len(dataset), n_train, n_val))
    # TRAIN_DATASET, TEST_DATASET = torch.utils.data.random_split(dataset, [n_train, n_val])

    TRAIN_DATASET = RegSupportDataset(root=os.path.join(root, "train"), npoints=args.npoint, split='train_val', normal_channel=args.normal, n_class=NUM_CLASSES, f_cols=f_cols)
    TEST_DATASET = RegSupportDataset(root=os.path.join(root, "val"), npoints=args.npoint, split='train_val', normal_channel=args.normal, n_class=NUM_CLASSES, f_cols=f_cols)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=args.batch_size, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=args.batch_size)
    weights = torch.Tensor(TRAIN_DATASET.label_weights).cuda()
    print("weights: ", weights)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))
    num_classes = len(seg_classes)   # 1
    seg_label_to_cat = {}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat
    num_part = len(seg_label_to_cat)
    print("num_part: ", num_part, " num_classes: ", num_classes)
    args.normal = True
    max_points = 1000
    classifier = MODEL.get_model2(num_part, normal_channel=args.normal, num_categories=num_classes, additional_channel=f_cols-3, max_points=max_points).cuda()
    # classifier = MODEL.get_model(NUM_CLASSES, f_cols).cuda()
    # criterion = MODEL.get_loss().cuda()
    criterion = MODEL.MultiTaskLoss2().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print(str(experiment_dir))
    
    best_iou = 0
    best_acc = 0.0
    min_loss = 1e20
    try:
        # checkpoint = torch.load(str(experiment_dir) + '/checkpoints/model50.pth')
        checkpoint = torch.load("./log/part_seg/pointnet2_part_seg_msg_support_12_normal_4w_sample_batch4_2class/checkpoints/best_acc_model.pth")
        if 'class_avg_iou' in checkpoint.keys():
            best_iou = checkpoint['class_avg_iou']
            print("load best iou: ", best_iou)
        if 'best_acc' in checkpoint.keys():
            best_acc = checkpoint['best_acc']
        # if 'min_loss' in checkpoint.keys():
        #     min_loss = checkpoint['min_loss']
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)
        log_string('Use pretrain model')
        print(start_epoch)
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        # 只收集需要训练的参数
        trainable_params = filter(lambda p: p.requires_grad, classifier.parameters())

        # 初始化优化器（示例使用Adam）
        optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
        # optimizer = torch.optim.Adam(
        #     classifier.parameters(),
        #     lr=args.learning_rate,
        #     betas=(0.9, 0.999),
        #     eps=1e-08,
        #     weight_decay=args.decay_rate
        # )
    else:
        # 只收集需要训练的参数
        trainable_params = filter(lambda p: p.requires_grad, classifier.parameters())
        optimizer = torch.optim.SGD(trainable_params, lr=args.learning_rate, momentum=0.9)
        # optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    
    best_epoch = start_epoch
    global_epoch = 0

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        # 但冻结的BatchNorm层需保持评估模式
        for m in classifier.modules():
            if isinstance(m, torch.nn.BatchNorm1d) and not m.weight.requires_grad:
                m.eval()

        for i, (points, gt_coords, gt_cnts, cls_tooth) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points = points.data.numpy()
            gt_coords = gt_coords.data.numpy()
            if np.random.choice([0, 1]):
                if args.normal:
                    # points[:, :, :6] = provider.rotate_point_cloud_z_with_normal(points[:, :, :6]) 
                    points[:, :, :6], gt_coords = provider.rotate_point_cloud_z_with_normal_gt_coords(points[:, :, :6], gt_coords)
                    # points[:, :, :6] = provider.rotate_point_cloud_with_normal(points[:, :, :6])
                else:
                    points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
                    # points[:, :, :3] = provider.rotate_point_cloud(points[:, :, :3]) 
            # print(points.shape, gt_coords.shape)
            points = torch.Tensor(points) 
            gt_coords = torch.Tensor(gt_coords) 
            gt_cnts =torch.Tensor(gt_cnts / max_points).float().cuda()
            points, gt_coords = points.float().cuda(), gt_coords.float().cuda()
            points = points.transpose(2, 1)
            # print(points.shape, points[0])
            pred_coords, pred_probs, pred_count = classifier(points, to_categorical(cls_tooth.long().cuda(), num_classes))
             
            # print("ori_xyz", ori_xyz.shape,  points[:, :3, :].shape)
            loss = criterion(pred_coords, pred_probs, pred_count, gt_coords, gt_cnts, weights) 
            loss.backward()
            optimizer.step()
            writer.add_scalar("train/loss", loss, epoch)
            loss_sum += loss

        log_string('Training mean loss: %f' % (loss_sum / num_batches))

        if (epoch + 1) % 50 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model' + str(epoch + 1) + '.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch + 1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            loss_sum = 0
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, gt_coords, gt_cnts, cls_tooth) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                # points = points.data.numpy()
                # points = torch.Tensor(points)
                 
                points, gt_coords = points.float().cuda(), gt_coords.float().cuda()
                gt_cnts = (gt_cnts / max_points).float().cuda()
                points = points.transpose(2, 1)

                pred_coords, pred_probs, pred_count= classifier(points, to_categorical(cls_tooth.long().cuda(), num_classes))
                 
                loss = criterion(pred_coords, pred_probs, pred_count, gt_coords, gt_cnts, weights) 
                loss_sum += loss 
  
            mLoss = loss_sum / float(num_batches)
            log_string('eval mean loss: %f' % (mLoss)) 
 
            if mLoss < min_loss:
                min_loss = mLoss
                best_epoch = epoch + 1
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/min_loss_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'min_loss': mLoss,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....') 
            log_string('Best loss: %f, best_epoch: %d' % min_loss, best_epoch)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
