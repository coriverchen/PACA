
import os

import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch.optim as optim
import torch.nn as nn
import argparse
import torchvision
from torchvision import transforms
import torchvision.models as models
from advertorch.utils import NormalizeByChannelMeanStd

from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='training transformer')
parser.add_argument('--max_epochs', type=int, default=80, help='max training epochs')
parser.add_argument('--seed', type=int, default=2020, help='random seed')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', default='cifar10', help='cifar10 | cifar100 | svhn | ile')
parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', default='resnet_ilsvrc', help='resnet | densenet | resnet_ilsvrc')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--adv_type', default='DDNL2', help='FGSM | BIM | DeepFool | CWL2')
parser.add_argument('--save_dir', default='features', help='where to save output features')
args = parser.parse_args()
print(args)

if args.dataset == 'cifar10':
    args.num_classes = 10
if args.dataset == 'imagenet':
    args.num_classes = 1000

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# %%

in_transform = transforms.Compose([transforms.ToTensor()])
advname = args.adv_type
writer = SummaryWriter(log_dir='log2/image_srncovnet_two_stream_'+advname)


datasets =  torchvision.datasets.ImageFolder('image_test_disjoint/'+advname, transform=in_transform)
dataset_size = len(datasets)
# import ipdb
# ipdb.set_trace()
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
split2 = int(np.floor(0.9 * dataset_size))
np.random.seed(args.seed)
np.random.shuffle(indices)
train_indices, val_indices,test_indices = indices[split:split2], indices[:split], indices[split2:]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(datasets, batch_size=16,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(datasets, batch_size=16,
                                                sampler=valid_sampler)

# test_loader = torch.utils.data.DataLoader(datasets, batch_size=64,
#                                                 sampler=test_sampler)

test_datasets =  torchvision.datasets.ImageFolder('image_test/'+advname, transform=in_transform)
test_indices = list(range(len(test_datasets)))
test_sampler = SubsetRandomSampler(test_indices)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=16,
                                                sampler=test_sampler)

import net

model = net.two_stream_srncovet()

state_dict = torch.load('./ckpt/SRN_covet_single_image_fine_cwl21.pkl')
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    # name = k[7:] # remove `module.
    name = 'spatial.' + k
    new_state_dict[name] = v
    name2 = 'gradient.' + k
    new_state_dict[name2] = v

# load params
model.load_state_dict(new_state_dict)



use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()
    print ('USE GPU')
else:
    print ('USE CPU - (please modify [self.const_kernel.cuda()] as [self.const_kernel])')


global is_best,best_epoch
is_best =0
best_epoch = 0    
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 70, 150], gamma=0.1)

normalize = NormalizeByChannelMeanStd(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model2 = models.resnet34(pretrained=True)
model2 = nn.Sequential(normalize, model2)
model2 = model2.cuda()
model2 = model2.eval()
model2.retain_graph = True


# %%
def train(epoch):
    model.train()
    lr_scheduler.step(epoch=epoch)
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
        writer.add_scalar('learning rate',param_group['lr'], epoch)
        break

    # adjust_learning_rate(optimizer,epoch)
    global is_best,best_epoch
    train_loss = 0
    train_acc = 0
    for batch_idx, data in enumerate(train_loader):
        images,labels = data     
        if use_gpu:
            images, labels= images.cuda(), labels.cuda()   
        optimizer.zero_grad()
        images = torch.autograd.Variable(images, requires_grad=True)
        y_logit = model2(images)
        _, pred_class = torch.max(y_logit, 1)
        loss = criterion(y_logit, pred_class)
        gradient = torch.autograd.grad(loss, images)[0]
        gradient = torch.abs(gradient).detach().cuda()
        # fuse_input = torch.cat((images,gradient),1)

        mini_out = model(images,gradient)
        mini_loss  =criterion(mini_out, labels.long())
        mini_loss.backward()
        optimizer.step() 
        
        _, pred = torch.max(mini_out, 1)
        acc = (pred.data == labels.long()).sum()  
        train_loss += float(mini_loss)
        train_acc += float(acc)
        if batch_idx % 10== 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Acc:{:.6f}'.format(
                epoch, batch_idx * len(labels), len(train_loader.sampler),
                100. * batch_idx / len(train_loader),
                mini_loss / len(labels),float(acc) / len(labels)))

    print('====> Epoch: {} Average loss: {:.6f} Average accuracy: {:.6f}'.format(
          str(epoch), train_loss / (len(train_loader.sampler)), train_acc / (len(train_loader.sampler))))
    # torch.save(model.state_dict(), 'pretrained/model'+epoch+'.pkl')
    writer.add_scalar('train_loss', train_loss / (len(train_loader.sampler)), epoch)
    writer.add_scalar('train_acc', train_acc / (len(train_loader.sampler)), epoch)
    torch.cuda.empty_cache()
    test_loss_epoch, test_acc_epoch = valid(epoch)
    if test_acc_epoch>is_best:
        is_best = test_acc_epoch
        best_epoch = epoch
        torch.save(model.state_dict(), './ckpt/image_srncovnet_twostream_'+ advname +'.pkl')
    torch.cuda.empty_cache()


def valid(epoch):
    model.eval()
    test_loss = 0
    test_acc = 0
    for i, data in enumerate(validation_loader):
        images,labels = data     
        if use_gpu:
            images, labels= images.cuda(), labels.cuda() 
        images = torch.autograd.Variable(images, requires_grad=True)
        y_logit = model2(images)
        _, pred_class = torch.max(y_logit, 1)
        loss = criterion(y_logit, pred_class)
        gradient = torch.autograd.grad(loss, images)[0]
        gradient = torch.abs(gradient).detach().cuda()
        # fuse_input = torch.cat((images,gradient),1)
        with torch.no_grad():
            mini_out = model(images,gradient)
            mini_loss  =criterion(mini_out, labels.long())
            _, pred = torch.max(mini_out, 1)
            acc = (pred.data == labels.long()).sum()  
            test_loss += float(mini_loss)
            test_acc += float(acc)        
    print('====> Epoch: {} Valid Set Average loss: {:.4f} Average accuracy: {:.4f}'.format(
         epoch, test_loss / (len(validation_loader.sampler)), test_acc / (len(validation_loader.sampler))))
    writer.add_scalar('test_loss', test_loss / (len(validation_loader.sampler)), epoch)
    writer.add_scalar('test_acc', test_acc / (len(validation_loader.sampler)), epoch)      
    return  test_loss / (len(validation_loader.sampler)),test_acc / (len(validation_loader.sampler))



def test(epoch):
    model.eval()
    test_loss = 0
    test_acc = 0
    for i, data in enumerate(test_loader):
        images,labels = data     
        if use_gpu:
            images, labels= images.cuda(), labels.cuda() 
        images = torch.autograd.Variable(images, requires_grad=True)
        y_logit = model2(images)
        _, pred_class = torch.max(y_logit, 1)
        loss = criterion(y_logit, pred_class)
        gradient = torch.autograd.grad(loss, images)[0]
        gradient = torch.abs(gradient).detach().cuda()
        # fuse_input = torch.cat((images,gradient),1)
        with torch.no_grad():
            mini_out = model(images,gradient)
            mini_loss  =criterion(mini_out, labels.long())
            _, pred = torch.max(mini_out, 1)
            acc = (pred.data == labels.long()).sum()  
            test_loss += float(mini_loss)
            test_acc += float(acc)        
    print('====> Epoch: {} Test Set Average loss: {:.4f} Average accuracy: {:.4f}'.format(
         epoch, test_loss / (len(test_loader)), test_acc / (len(test_loader))))
    return  test_loss / (len(test_loader)),test_acc / (len(test_loader))

# %%
for epoch in range(200):
    train(epoch)
model.load_state_dict(torch.load('./ckpt/image_PACA_'+ advname +'.pkl'))
model.eval()
test(0)
writer.close()
