from model import config
from model.fcos import FCOSDetector
import torch
from dataset.VOC_dataset import VOCDataset
import math, time
from dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=60, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
parser.add_argument("--height", type=int, default=800, help="height of each image")
parser.add_argument("--width", type=int, default=1333, help="width of each image")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0,1', help="number of cpu threads to use during batch generation")
parser.add_argument("--LR_INIT", type=float, default=1e-3, help="init learing rate")
parser.add_argument("--interval", type=int, default=5, help="How long to save the model")
parser.add_argument("--resume", type=bool, default=False, help="whether or Continue to train the model")
parser.add_argument("--resumed_path", type=str, default='', help="resmued trained path")
parser.add_argument("--saved_path", type=str, default="./checkpoint", help="saved path of trained model")
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
transform = Transforms()
resize_size = [opt.height, opt.width]
interval = opt.interval
train_dataset = VOCDataset(root_dir='./data/VOCdevkit/VOC2007',resize_size=resize_size,
                            split='trainval',use_difficult=False,is_train=True,augment=transform)

model = FCOSDetector(mode="training").cuda()
model = torch.nn.DataParallel(model)

saved_path = opt.saved_path
if not os.path.exists(saved_path):
    os.makedirs(saved_path)

# model.load_state_dict(torch.load('./checkpoint/model_100.pth'))
# model.load_state_dict(torch.load('/mnt/cephfs_new_wj/vc/zhangzhenghao/FCOS.Pytorch/output1/model_6.pth'))

BATCH_SIZE = opt.batch_size
EPOCHS = opt.epochs
#WARMPUP_STEPS_RATIO = 0.12
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                            collate_fn=train_dataset.collate_fn,
                                            num_workers=opt.n_cpu, worker_init_fn=np.random.seed(0))
print("total_images : {}".format(len(train_dataset)))
steps_per_epoch = len(train_dataset) // BATCH_SIZE
TOTAL_STEPS = steps_per_epoch * EPOCHS
WARMPUP_STEPS = 501
WARMUP_FACTOR = 1.0 / 3.0

GLOBAL_STEPS = 1
LR_INIT = opt.LR_INIT
LR_END = 2e-5
optimizer = torch.optim.SGD(model.parameters(),lr=LR_INIT, momentum=0.9, weight_decay=0.0001)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)

# def lr_func():
#     if GLOBAL_STEPS < WARMPUP_STEPS:
#         lr = GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT
#     else:
#         lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
#              (1 + math.cos((GLOBAL_STEPS - WARMPUP_STEPS) / (TOTAL_STEPS - WARMPUP_STEPS) * math.pi))
#         )
#     return float(lr)

lr_schedule = [20001, 27001, 32001]
def lr_func(step):
    lr = LR_INIT
    if step < WARMPUP_STEPS:
        # alpha = float(step) / WARMPUP_STEPS
        # warmup_factor = WARMUP_FACTOR * (1.0 - alpha) + alpha
        # lr = lr*warmup_factor
        lr = float(GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT)
    else:
        for i in range(len(lr_schedule)):
            if step < lr_schedule[i]:
                break
            lr *= 0.1
    return float(lr)


model.train()

init_epoch = 0
if opt.resume:
    checkpoint = torch.load(opt.resumed_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    initepoch = checkpoint['epoch'] + 1
    GLOBAL_STEPS = checkpoint['GLOBAL_STEPS']


for epoch in range(init_epoch, EPOCHS):
    base = 0.1
    lr = LR_INIT
    def update_lr(lr, base):
        lr = LR_INIT * base
        for param in optimizer.param_groups:
            param['lr'] = lr
        base *= 0.1
        return lr, base
    
    try:

        for epoch_step, data in enumerate(train_loader):

            batch_imgs, batch_boxes, batch_classes = data
            batch_imgs = batch_imgs.cuda()
            batch_boxes = batch_boxes.cuda()
            batch_classes = batch_classes.cuda()

            lr = lr_func(GLOBAL_STEPS)
            for param in optimizer.param_groups:
                param['lr']=lr
            if GLOBAL_STEPS < WARMPUP_STEPS:
                lr = float(GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT)
                for param in optimizer.param_groups:
                    param['lr'] = lr
            
            # if GLOBAL_STEPS in (20001, 27001):
            #     lr, base = update_lr(lr, base)
            
            start_time = time.time()

            optimizer.zero_grad()
            losses = model([batch_imgs, batch_boxes, batch_classes])
            loss = losses[-1]
            loss.mean().backward()
            optimizer.step()
            end_time = time.time()
            cost_time = int((end_time - start_time) * 1000)
            if config.DefaultConfig.use_gl:
                print("global_steps:%d epoch:%d steps:%d/%d qfl_loss:%.6f bbox_loss:%.6f dfl_loss:%.6f cost_time:%dms lr=%.6e total_loss:%.6f" % \
                        (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
                        losses[2].mean(), cost_time, lr, loss.mean()))
            else:
                if len(losses) == 4:
                    print(
                        "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.6f cnt_loss:%.6f reg_loss:%.6f cost_time:%dms lr=%.6e total_loss:%.6f" % \
                        (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
                        losses[2].mean(), cost_time, lr, loss.mean()))
                else:
                    print(
                        "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.6f  reg_loss:%.6f cost_time:%dms lr=%.6e total_loss:%.6f" % \
                        (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, 
                        losses[0].mean(), losses[1].mean(), cost_time, lr, loss.mean()))

            GLOBAL_STEPS += 1

    except Exception as e:
        print(e)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'GLOBAL_STEPS': GLOBAL_STEPS,
        }
        torch.save(checkpoint, os.path.join(saved_path, "model_{}_{}_{}_{}.pth".format(BATCH_SIZE, resize_size[0],resize_size[1], epoch + 1)))

    if (epoch + 1) % interval == 0 or (epoch + 1) == EPOCHS:
        torch.save(model.state_dict(), os.path.join(saved_path, "model_{}_{}_{}_{}.pth".format(BATCH_SIZE, resize_size[0],resize_size[1], epoch + 1)))














