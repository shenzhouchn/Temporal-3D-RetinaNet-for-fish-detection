import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

from tensorboardX import SummaryWriter

import time
import collections

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=25)

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    # create samplers for both training and validation 
    # using muti CPU cores to accelerate data loading

    sampler_train1 = torch.utils.data.SequentialSampler(dataset_train)
    sampler_train2 = torch.utils.data.BatchSampler(sampler_train1, batch_size=1, drop_last=True)
    dataloader_train = DataLoader(dataset_train, num_workers=10, collate_fn=collater, batch_sampler=sampler_train2)

    sampler_val1 = torch.utils.data.SequentialSampler(dataset_val)
    sampler_val2 = torch.utils.data.BatchSampler(sampler_val1, batch_size=1, drop_last=True)
    dataloader_val = DataLoader(dataset_val, num_workers=10, collate_fn=collater, batch_sampler=sampler_val2)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    # ADAM optimizer
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    # using tensorboardX to show training process
    writer = SummaryWriter('log')

    iter_sum = 0
    time_sum = 0
    frame_num = 8

    for epoch_num in range(parser.epochs):

        # only work for frame_num > 8
        frame_list = collections.deque(maxlen=frame_num)
        anno_list = collections.deque(maxlen=frame_num)

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for index, data in enumerate(dataloader_train):
            try:

                frame_list.append(data['img'])
                anno_list.append(data['annot'])

                # if frame_num != 32:
                if index < 31:
                    continue
                if index >= 697 and index <= 697+32:
                    continue

                # real_frame is the frame we used for fish detection
                # It's the last frame in the batch group
                real_frame = frame_list[-1]

                # the annotation for real_frame
                annot = anno_list[-1]

                # drop useless frames
                data['img'] = torch.cat(list(frame_list),dim=0)

                optimizer.zero_grad()

                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), real_frame.cuda().float(), annot.cuda().float()])
                
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                writer.add_scalar('loss_hist',np.mean(loss_hist),iter_sum)
                writer.add_scalar('classification_loss',float(classification_loss),iter_sum)
                writer.add_scalar('regression_loss',float(regression_loss),iter_sum)
                writer.add_scalar('loss',float(loss),iter_sum)

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, index, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
                iter_sum = iter_sum + 1
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            # evaluate coco
            coco_eval.evaluate_coco(dataset_val, dataloader_val, retinanet, frame_num)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, 'checkpoint/{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()

    torch.save(retinanet, 'save/model_final.pt')

    writer.close()


if __name__ == '__main__':
    main()
