
# coding: utf-8

# In[ ]:


# Training script for tiny-imagenet.
# Again, this script has a lot of bugs everywhere.
import argparse
import os
import shutil

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import losses

from models.discriminators.snresnet64 import Omniglot_Discriminator, VGG_Discriminator, Animal_Discriminator
from models.generators.resnet64 import Omniglot_Generator, VGG_Generator, Animal_Generator

from dataloader import omniglot_data_loader, vgg_data_loader, img_dataloder, celeba_data_loader
import utils
from torch.nn.utils import spectral_norm
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter


dev = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_ROTATIONS = 4

# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L5-L15
def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L18-L26
class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def decay_lr(opt, max_iter, start_iter, initial_lr):
    """Decay learning rate linearly till 0."""
    coeff = -initial_lr / (max_iter - start_iter)
    for pg in opt.param_groups:
        pg['lr'] += coeff


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_version", type=str, default="3-shot_vgg_0.0002")
    parser.add_argument('--dataset', type=str, default="vgg", help="omniglot or vgg")
    parser.add_argument('--dataset_root', type=str, default="/home/userB/yonggyukim/data")
    parser.add_argument('--n_shot', type=str, default="3-shot")
    
    parser.add_argument('--resize_size', type=int, default=84)
    parser.add_argument('--crop_size', type=int, default=64)

    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Initial learning rate of Adam. default: 0.0002')
    parser.add_argument('--lr_decay_start', '-lds', type=int, default=50000,
                        help='Start point of learning rate decay. default: 50000')

    # Dataset configuration
    parser.add_argument('--cGAN', default=True, action='store_true',
                        help='to train cGAN, set this ``True``. default: False')
    parser.add_argument('--data_root', type=str, default='tiny-imagenet-200',
                        help='path to dataset root directory. default: tiny-imagenet-200')
    parser.add_argument('--batch_size', '-B', type=int, default=64,
                        help='mini-batch size of training data. default: 64')
    parser.add_argument('--eval_batch_size', '-eB', default=None,
                        help='mini-batch size of evaluation data. default: None')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for training data loader. default: 8')
    # Generator configuration
    parser.add_argument('--gen_num_features', '-gnf', type=int, default=32,
                        help='Number of features of generator (a.k.a. nplanes or ngf). default: 64')
    parser.add_argument('--gen_dim_z', '-gdz', type=int, default=128,
                        help='Dimension of generator input noise. default: 128')
    parser.add_argument('--gen_bottom_width', '-gbw', type=int, default=7,
                        help='Initial size of hidden variable of generator. default: 4')
    parser.add_argument('--gen_distribution', '-gd', type=str, default='normal',
                        help='Input noise distribution: normal (default) or uniform.')
    # Discriminator (Critic) configuration
    parser.add_argument('--dis_arch_concat', '-concat', default=False, action='store_true',
                        help='If use concat discriminator, set this true. default: False')
    parser.add_argument('--dis_emb', type=int, default=128,
                        help='Parameter for concat discriminator. default: 128')
    parser.add_argument('--dis_num_features', '-dnf', type=int, default=32,
                        help='Number of features of discriminator (a.k.a nplanes or ndf). default: 64')
    # Optimizer settings
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='beta1 (betas[0]) value of Adam. default: 0.0')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 (betas[1]) value of Adam. default: 0.9')
    # Training setting
    parser.add_argument('--seed', type=int, default=46,
                        help='Random seed. default: 46 (derived from Nogizaka46)')
    parser.add_argument('--max_iteration', '-N', type=int, default=100000,
                        help='Max iteration number of training. default: 100000')
    parser.add_argument('--n_dis', type=int, default=5,
                        help='Number of discriminator updater per generator updater. default: 5')
    parser.add_argument('--num_classes', '-nc', type=int, default=0,
                        help='Number of classes in training data. No need to set. default: 0')
    parser.add_argument('--loss_type', type=str, default='hinge',
                        help='loss function name. hinge (default) or dcgan.')
    parser.add_argument('--relativistic_loss', '-relloss', default=False, action='store_true',
                        help='Apply relativistic loss or not. default: False')
    parser.add_argument('--calc_FID', default=False, action='store_true',
                        help='If calculate FID score, set this ``True``. default: False')
    # Log and Save interval configuration
    parser.add_argument('--results_root', type=str, default='results',
                        help='Path to results directory. default: results')
    parser.add_argument('--no_tensorboard', action='store_true', default=False,
                        help='If you dislike tensorboard, set this ``False``. default: True')
    parser.add_argument('--no_image', action='store_true', default=False,
                        help='If you dislike saving images on tensorboard, set this ``True``. default: False')
    parser.add_argument('--checkpoint_interval', '-ci', type=int, default=1000,
                        help='Interval of saving checkpoints (model and optimizer). default: 1000')
    parser.add_argument('--log_interval', '-li', type=int, default=100,
                        help='Interval of showing losses. default: 100')
    parser.add_argument('--eval_interval', '-ei', type=int, default=100,
                        help='Interval for evaluation (save images and FID calculation). default: 1000')
    parser.add_argument('--n_eval_batches', '-neb', type=int, default=100,
                        help='Number of mini-batches used in evaluation. default: 100')
    parser.add_argument('--n_fid_images', '-nfi', type=int, default=50,
                        help='Number of images to calculate FID. default: 5000')
    parser.add_argument('--test', default=False, action='store_true',
                        help='If test this python program, set this ``True``. default: False')
    # Resume training
    parser.add_argument('--args_path', default=None, help='Checkpoint args json path. default: None')
    parser.add_argument('--gen_ckpt_path', '-gcp', default=None,
                        help='Generator and optimizer checkpoint path. default: None')
    parser.add_argument('--dis_ckpt_path', '-dcp', default=None,
                        help='Discriminator and optimizer checkpoint path. default: None')
    args = parser.parse_args()
    return args


def sample_from_data(args, device, data_loader):
    real, y = next(data_loader)
    if real.size(0) < args.batch_size:
        real, y = next(data_loader)
    real, y = real.to(device), y.to(device)
    if not args.cGAN:
        y = None
    return real, y


def sample_from_gen(args, device, num_classes, gen):
    z = utils.sample_z(
        args.batch_size, args.gen_dim_z, device, args.gen_distribution
    )
    if args.cGAN:
        pseudo_y = utils.sample_pseudo_labels(
            num_classes, args.batch_size, device
        )
    else:
        pseudo_y = None

    fake = gen(z, pseudo_y)
    return fake, pseudo_y, z


def pick_fixed_img(args, train_loader, img_num):
    img_list = []
    label_list = []

    for i in range(8):
        x_data, y_data = sample_from_data(args, dev, train_loader)
        for j in range(x_data.size(0)):
            img_list.append(x_data[j])
            label_list.append(y_data[j])

    img_list = img_list[0: img_num]
    label_list = label_list[0: img_num]

    return img_list, label_list


def directory_path(args):
    output = "output"
    weight_path = os.path.join(output, args.exp_version, 'weight')
    img_path = os.path.join(output, args.exp_version, 'img')
    loss_path = os.path.join(output, args.exp_version, 'loss')

    if os.path.exists(weight_path) is False:
        os.makedirs(weight_path)
    if os.path.exists(img_path) is False:
        os.makedirs(img_path)
    if os.path.exists(loss_path) is False:
        os.makedirs(loss_path)

    return weight_path, img_path, loss_path


def data_loader2(args):
    root_path = args.dataset_root
    data_root = os.path.join(root_path, args.dataset)
    print(data_root)
    if args.dataset == "omniglot":
        train_loader, s_dlen = img_dataloder(
            args=args,
            root=data_root
        )
        print("omniglot data_loader")
        num_classes = 1623
    elif args.dataset == "vgg" or args.dataset == "cub":
        train_loader, s_dlen = img_dataloder(
            args=args,
            root=data_root
        )
        print("vgg data_loader")
        if args.dataset == "vgg":
            num_classes = 2300
        elif args.dataset == "cub":
            num_classes = 200
    elif args.dataset == "animal":
        train_loader, s_dlen = img_dataloder(
            args=args,
            root=data_root
        )
        print("animal data_loader")
        num_classes = 147
#     elif args.dataset == "celeba":
#         root_path = args.dataset_root
#         data_root = os.path.join(root_path, args.dataset)
#         print(data_root)
#         train_loader, s_dlen, num_classes = celeba_data_loader(
#             root=data_root,
#             batch_size=64)
    else:
        raise Exception("Enter omniglot or vgg or animal or celeba")

    train_loader = iter(utils.cycle(train_loader))
    return train_loader, s_dlen, num_classes


def select_model(args, _n_cls):
    print("selecting model")
    if args.dataset == "omniglot":
        gen = Omniglot_Generator(
            args.gen_num_features, args.gen_dim_z, bottom_width=7, activation=F.relu,
            num_classes=_n_cls, distribution=args.gen_distribution).to(dev)
        dis = Omniglot_Discriminator(args.dis_num_features, _n_cls, F.relu).to(dev)
    elif args.dataset == "vgg" or args.dataset == "cub" or args.crop_size==64:
        gen = VGG_Generator(
            args.gen_num_features * 2, args.gen_dim_z, bottom_width=4, activation=F.relu,
            num_classes=_n_cls, distribution=args.gen_distribution).to(dev)
        dis = VGG_Discriminator(args.gen_num_features * 2, _n_cls, F.relu).to(dev)
        print(111)
#     elif args.dataset == "celeba":
#         gen = VGG_Generator(
#             args.gen_num_features * 2, args.gen_dim_z, bottom_width=4, activation=F.relu,
#             num_classes=_n_cls, distribution=args.gen_distribution).to(dev)
#         dis = VGG_Discriminator(args.gen_num_features * 2, _n_cls, F.relu).to(dev)
    elif args.dataset == "animal":
        gen = Animal_Generator(
            args.gen_num_features * 2, args.gen_dim_z, bottom_width=4, activation=F.relu,
            num_classes=_n_cls, distribution=args.gen_distribution).to(dev)
        dis = Animal_Discriminator(args.gen_num_features * 2, _n_cls, F.relu).to(dev)
    else:
        raise Exception("Enter model omniglot or vgg or animal")

    return gen, dis



def main():
    args = get_args()
    weight_path, img_path, loss_path = directory_path(args)
    writer = SummaryWriter(loss_path)

    # CUDA setting
    if not torch.cuda.is_available():
        raise ValueError("Should buy GPU!")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True

    # dataloading
    train_loader, s_dlen, _n_cls = data_loader2(args)

    fixed_z = torch.randn(200, 10, 128)
    fixed_img_list, fixed_label_list = pick_fixed_img(args, train_loader, 200)
    
    fixed_img_list2 = []
    fixed_label_list2 = []
    for i in range(len(fixed_label_list)):
        if fixed_label_list[i].item() != _n_cls:
            fixed_label_list2.append(fixed_label_list[i])
            fixed_img_list2.append(fixed_img_list[i])
    
    # initialize model
    gen, dis = select_model(args, _n_cls)

    opt_gen = optim.Adam(gen.parameters(), args.lr, (args.beta1, args.beta2))
    opt_dis = optim.Adam(dis.parameters(), args.lr, (args.beta1, args.beta2))

    criterion = nn.CrossEntropyLoss()

    bs = args.batch_size
    # Training loop
    print(2)
    for n_iter in tqdm.tqdm(range(0, args.max_iteration)):

        if n_iter >= args.lr_decay_start:
            decay_lr(opt_gen, args.max_iteration, args.lr_decay_start, args.lr)
            decay_lr(opt_dis, args.max_iteration, args.lr_decay_start, args.lr)

        # ==================== Beginning of 1 iteration. ====================
        _l_g = .0
        cumulative_loss_dis = .0

        opt_gen.zero_grad()
        opt_dis.zero_grad()

        real_images, real_y = sample_from_data(args, dev, train_loader)
        fake_images, pseudo_y, _ = sample_from_gen(args, dev, _n_cls, gen)

        real_y = real_y.cuda()
        pseudo_y = pseudo_y.cuda()

        real_labels = torch.FloatTensor(real_y.size(0), _n_cls+1).cuda()
        fake_labels = torch.FloatTensor(real_y.size(0), _n_cls+1).cuda()

        real_labels.zero_()
        fake_labels.zero_()

        real_labels.scatter_(1, real_y.unsqueeze(1), 1)
        fake_labels.scatter_(1, pseudo_y.unsqueeze(1), 1)

        rotated_bs = bs // 2
        num_rot_examples = rotated_bs // NUM_ROTATIONS

        all_features, all_labels = utils.merge_with_rotation_data(
            real_images, fake_images, real_labels[:, :_n_cls], fake_labels[:, :_n_cls], num_rot_examples)

        """============================= Discriminator Forward ============================="""
        
        for dis_iter in range(2):
            d_predictions_d, d_logits_d, rot_logits_d, aux_logits_d, is_label_available_d=dis.discriminator_with_additonal_heads(x=all_features.detach(), y=all_labels.detach())

            expected_batch_size = 2 * bs
            expected_batch_size += 2 * (NUM_ROTATIONS - 1) * num_rot_examples

            if d_logits_d.shape[0] != expected_batch_size:
                raise ValueError("Batch size unexpected: got %r expected %r" % (
                    d_logits_d.shape[0], expected_batch_size))

            prob_real_d, prob_fake_d = torch.chunk(d_predictions_d, 2, dim=0)
            prob_real_d, prob_fake_d = prob_real_d[:bs], prob_fake_d[:bs]

            logits_real_d, logits_fake_d = torch.chunk(d_logits_d, 2, dim=0)
            logits_real_d, logits_fake_d = logits_real_d[:bs], logits_fake_d[:bs]

            d_loss, _, _, _ = losses.hinge_losses(
                d_real=prob_real_d, d_fake=prob_fake_d,
                d_real_logits=logits_real_d, d_fake_logits=logits_fake_d)


            rot_real_logits, _ = torch.chunk(rot_logits_d, 2, dim=0)
            rot_real_logits = rot_real_logits[-rotated_bs:]

            labels_rotated = torch.tensor(list(range(NUM_ROTATIONS))).repeat(num_rot_examples)
            real_loss = criterion(rot_real_logits, labels_rotated)
            d_loss += real_loss * 1.0

            real_aux_logits, _ = torch.chunk(aux_logits_d, 2, dim=0)
            real_aux_logits = real_aux_logits[:bs]

            is_label_available, _ = torch.chunk(is_label_available_d, 2, dim=0)
            is_label_available = is_label_available.squeeze(1)[:bs].unsqueeze(1)

            class_loss_real = losses.weighted_cross_entropy(
                real_labels[:, :_n_cls], real_aux_logits, weights=is_label_available)

            d_loss += class_loss_real * 1.0

            opt_gen.zero_grad()
            opt_dis.zero_grad()
            d_loss.backward()
            opt_dis.step()
        
        """=============================== Generator Forward ============================="""
        
        fake_images, pseudo_y, _ = sample_from_gen(args, dev, _n_cls, gen)
        pseudo_y = pseudo_y.cuda()

        fake_labels = torch.FloatTensor(real_y.size(0), _n_cls+1).cuda()
        fake_labels.zero_()
        fake_labels.scatter_(1, pseudo_y.unsqueeze(1), 1)

        rotated_bs = bs // 2
        num_rot_examples = rotated_bs // NUM_ROTATIONS

        all_features, all_labels = utils.merge_with_rotation_data(
            real_images, fake_images, real_labels[:, :_n_cls], fake_labels[:, :_n_cls], num_rot_examples)
        
        d_predictions, d_logits, rot_logits, aux_logits, is_label_available =  dis.discriminator_with_additonal_heads(x=all_features,
                                                                                                                      y=all_labels)

        if d_logits.shape[0] != expected_batch_size:
            raise ValueError("Batch size unexpected: got %r expected %r" % (
                d_logits.shape[0], expected_batch_size))

        prob_real, prob_fake = torch.chunk(d_predictions, 2, dim=0)
        prob_real, prob_fake = prob_real[:bs], prob_fake[:bs]

        logits_real, logits_fake = torch.chunk(d_logits, 2, dim=0)
        logits_real, logits_fake = logits_real[:bs], logits_fake[:bs]

        _, _, _, g_loss = losses.hinge_losses(
            d_real=prob_real, d_fake=prob_fake,
            d_real_logits=logits_real, d_fake_logits=logits_fake)

        _, rot_fake_logits = torch.chunk(rot_logits, 2, dim=0)
        rot_fake_logits = rot_fake_logits[-rotated_bs:]

        labels_rotated = torch.tensor(list(range(NUM_ROTATIONS))).repeat(num_rot_examples)
        fake_loss = criterion(rot_fake_logits, labels_rotated)

        g_loss += fake_loss * 0.2
        
        rot_real_pred = torch.argmax(rot_real_logits, dim=1, keepdim=True)
        rot_fake_pred = torch.argmax(rot_fake_logits, dim=1, keepdim=True)

        accuracy_real = (labels_rotated == rot_real_pred).sum().item() / rot_real_pred.size(0)
        accuracy_fake = (labels_rotated == rot_fake_pred).sum().item() / rot_fake_pred.size(0)
        
        opt_gen.zero_grad()
        opt_dis.zero_grad()
        g_loss.backward()
        opt_gen.step()

        writer.add_scalar('loss/total_G_loss', g_loss.item(), n_iter)
        writer.add_scalar('loss/total_D_loss', d_loss.item(), n_iter)
        writer.add_scalar('loss/real_loss', real_loss.item(), n_iter)
        writer.add_scalar('loss/fake_loss', fake_loss.item(), n_iter)
        writer.add_scalar('loss/G_loss', g_loss.item() - fake_loss.item(), n_iter)
        writer.add_scalar('loss/D_loss', d_loss.item() - real_loss.item() -
                          class_loss_real.item(), n_iter)

        writer.add_scalar('accuracy/real', accuracy_real, n_iter)
        writer.add_scalar('accuracy/fake', accuracy_fake, n_iter)
        writer.add_scalar("loss/class_loss_real", class_loss_real)
        writer.add_scalar("label_frac", torch.mean(is_label_available))

        # ==================== End of 1 iteration. ====================

        if n_iter % args.log_interval == 0:
            tqdm.tqdm.write(
                'iteration: {:07d}/{:07d}, loss gen: {:05f}, loss dis {:05f}'.format(
                    n_iter, args.max_iteration, g_loss.item(), d_loss.item()))

        if n_iter % args.checkpoint_interval == 0:
            #Save checkpoints!
            utils.save_checkpoints(args, n_iter, gen, opt_gen, dis, opt_dis, weight_path)
            utils.save_img(fixed_img_list2, fixed_label_list2, fixed_z, gen,
                           32, 28, img_path, n_iter, device=dev)
    if args.test:
        shutil.rmtree(args.results_root)


if __name__ == '__main__':
    main()

