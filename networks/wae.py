
import argparse
import math
import os
from tabnanny import verbose
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(123)

parser = argparse.ArgumentParser(description='PyTorch MNIST WAE-MMD')
parser.add_argument('-batch_size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 100)')

parser.add_argument('-epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')

parser.add_argument('-lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')

parser.add_argument('-dim_h', type=int, default=128,
                    help='hidden dimension (default: 128)')

parser.add_argument('-n_z', type=int, default=512,
                    help='hidden dimension of z (default: 512)')

parser.add_argument('-LAMBDA', type=float, default=10,
                    help='regularization coef MMD term (default: 10)')

parser.add_argument('-n_channel', type=int, default=1,
                    help='input channels (default: 1)')
parser.add_argument('-sigma', type=float, default=1,
                    help='variance of hidden dimension (default: 1)')

parser.add_argument('-img_width', type=int, default=256,
                    help='The width of the input images (default: 256)')

parser.add_argument('-img_height', type=int, default=256,
                    help='The width of the input images (default: 256)')

parser.add_argument('-img_channel', type=int, default=3,
                    help='The number of the channels of input images (default: 3)')

parser.add_argument('-train_data_root', type=str, default="./data/semantic256x256_test",
                    help='The path to the folder containing train images')

parser.add_argument('-test_data_root', type=str, default="./data/semantic256x256_test",
                    help='The path to the folder containing test images')

parser.add_argument('-tb_root', type=str, default="./logs/wae",
                    help='The path to the log directory of tensorboard')

parser.add_argument('-progress_root', type=str, default='./progress/reconst_images',
                    help='The path for saving reconstructed and sampled images.')

parser.add_argument('-checkpoint_root', type=str, default="./checkpoints/wae",
                    help='The path for saving checkpoints')

parser.add_argument('-checkpoint_load_root', type=str, default="",
                    help='Saved Checkpoint')



args = parser.parse_args()

trainset = ImageFolder(root=args.train_data_root,
                       transform=transforms.ToTensor())

testset = ImageFolder(root=args.test_data_root,
                      transform=transforms.ToTensor())

train_loader = DataLoader(dataset=trainset,
                          batch_size=args.batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=testset,
                         batch_size=args.batch_size,
                         shuffle=False)

IMAGE_SIZE = (args.img_channel, args.img_width, args.img_height)


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['leaky_relu', nn.LeakyReLU()],
        ['selu', nn.SELU()],
        ['none', nn.Identity()]
    ])[activation]


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding='valid',
                 pool_size=(2, 2),
                 activation='leaky_relu') -> None:
        super().__init__()

        self.activation = activation_func(activation)

        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            padding=padding,
            kernel_size=kernel_size,
        )

        self.batch_norm1 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            padding='same',
            kernel_size=kernel_size,
        )

        self.conv_short_cut = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=padding,
            kernel_size=kernel_size,
        )

        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.pooling = nn.MaxPool2d(pool_size)

    def forward(self, inputs):
        x = self.conv_in(inputs)
        x = self.activation(x)
        x = self.batch_norm1(x)

        x = self.conv1(x)
        x = self.activation(x)

        res = self.conv_short_cut(inputs)
        res = self.activation(res)

        x = x + res
        x = self.batch_norm2(x)

        x = self.pooling(x)

        return x


class ResidualBlockUpsample(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 scale_factor=2,
                 activation='leaky_relu',
                 padding='valid') -> None:
        super().__init__()
        self.activation = activation_func(activation)

        self.conv_in = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=(math.floor(kernel_size/2),
                     math.floor(kernel_size/2)) if padding == 'same' else 0
        )

        self.batch_norm1 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=((math.floor(kernel_size/2), math.floor(kernel_size/2))),
            kernel_size=kernel_size,
        )

        self.conv_short_cut = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(math.floor(kernel_size/2),
                     math.floor(kernel_size/2)) if padding == 'same' else 0

        )

        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.pooling = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, inputs):
        x = self.conv_in(inputs)
        x = self.activation(x)
        x = self.batch_norm1(x)

        x = self.conv1(x)
        x = self.activation(x)

        res = self.conv_short_cut(inputs)
        res = self.activation(res)
        x = x + res
        x = self.batch_norm2(x)

        x = self.pooling(x)

        return x


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            ResidualBlock(in_channels=3,
                          out_channels=16,
                          kernel_size=5),

            ResidualBlock(in_channels=16,
                          out_channels=32),

            ResidualBlock(in_channels=32,
                          out_channels=64),

            ResidualBlock(in_channels=64,
                          out_channels=64),

            ResidualBlock(in_channels=64,
                          out_channels=64)
        )
        self.fc = nn.Linear(6 * 6 * 64, self.n_z)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.proj = nn.Sequential(
            nn.Linear(self.n_z, 6 * 6 * 64),
            nn.ReLU()
        )

        self.main = nn.Sequential(
            ResidualBlockUpsample(
                in_channels=64,
                out_channels=64,
                padding='same',
            ),

            ResidualBlockUpsample(
                in_channels=64,
                out_channels=64,
            ),

            ResidualBlockUpsample(
                in_channels=64,
                out_channels=64,
            ),

            ResidualBlockUpsample(
                in_channels=64,
                out_channels=32,
            ),

            ResidualBlockUpsample(
                in_channels=32,
                out_channels=16,
                kernel_size=5
            ),

            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=3,
                kernel_size=3,
                padding=(math.floor(3/2), math.floor(3/2))
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.view(-1, 64, 6, 6)
        x = self.main(x)
        return x


def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats


def rbf_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 / scale
        res1 = torch.exp(-C * dists_x)
        res1 += torch.exp(-C * dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = torch.exp(-C * dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats += res1 - res2

    return stats


encoder, decoder = Encoder(args), Decoder(args)
criterion = nn.MSELoss()
start_epoch = 0
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")




encoder.train()
decoder.train()

# print(summary(encoder, (3,256,256)))
# print(summary(decoder, (args.n_z,)))


print("=========================================================")

print(f"cuda available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    encoder, decoder = encoder.cuda(), decoder.cuda()

one = torch.Tensor([1])
mone = one * -1

if torch.cuda.is_available():
    one = one.cuda()
    mone = mone.cuda()

# Optimizers
enc_optim = optim.Adam(encoder.parameters(), lr=args.lr)
dec_optim = optim.Adam(decoder.parameters(), lr=args.lr)

if args.checkpoint_load_root != '':
    cp = torch.load(args.checkpoint_load_root)
    print(f"checkpoint loaded from {args.checkpoint_load_root}")
    encoder.load_state_dict(cp['encoder_state_dict'])
    decoder.load_state_dict(cp['decoder_state_dict'])
    
    enc_optim.load_state_dict(cp['encoder_optim_state_dict'])
    dec_optim.load_state_dict(cp['decoder_optim_state_dict'])
    start_epoch = cp['epoch']
    current_time = cp['time']

enc_scheduler = StepLR(enc_optim, step_size=20, gamma=0.5, verbose=True, last_epoch = start_epoch + 1 if start_epoch != 0 else -1)
dec_scheduler = StepLR(dec_optim, step_size=20, gamma=0.5, verbose=True, last_epoch = start_epoch + 1 if start_epoch != 0 else -1)


tb_dir = f'{args.tb_root}{current_time}'
if not os.path.isdir(tb_dir):
    os.makedirs(tb_dir)
train_writer_total = SummaryWriter(f'{tb_dir}/train')
test_writer_total = SummaryWriter(f'{tb_dir}/test')


def capture_progress(epoch):
    batch_size = args.batch_size
    test_iter = iter(test_loader)
    test_data = next(test_iter)

    z_real = encoder(Variable(test_data[0]).cuda())
    reconst = decoder(z_real).cpu().view(
        batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2])
    sample = decoder(torch.randn_like(z_real)).cpu().view(
        batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2])

    progress_dir = args.progress_root

    if not os.path.isdir(progress_dir):
        os.makedirs(progress_dir)

    save_image(test_data[0].view(-1, IMAGE_SIZE[0], IMAGE_SIZE[1],
               IMAGE_SIZE[2]), f'{progress_dir}/wae_mmd_input.png')
    save_image(
        reconst.data, f'{progress_dir}/wae_mmd_images_%d.png' % (epoch + 1))
    save_image(
        sample.data, f'{progress_dir}/wae_mmd_samples_%d.png' % (epoch + 1))


def create_checkpoint(epoch):

    checkpoint_dir = args.checkpoint_root

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'encoder_optim_state_dict': enc_optim.state_dict(),
        'decoder_optim_state_dict': dec_optim.state_dict(),
        'epoch': epoch,
        'time': str(current_time)
    }, f'{checkpoint_dir}/checkpoint.pth')


last_total_loss = math.inf

for epoch in range(start_epoch + 1,args.epochs):
    step = 0
    recon_step_loss = 0
    mmd_step_loss = 0
    current_total_loss = 0
    capture_progress(epoch=epoch)
    with tqdm(train_loader) as tepoch:
        for (images, _) in train_loader:

            tepoch.set_description(f"Epoch {epoch + 1}")

            if torch.cuda.is_available():
                images = images.cuda()

            enc_optim.zero_grad()
            dec_optim.zero_grad()

            # ======== Train Generator ======== #

            batch_size = images.size()[0]

            z = encoder(images)
            x_recon = decoder(z)

            recon_loss = criterion(x_recon, images)

            # ======== MMD Kernel Loss ======== #

            z_fake = Variable(torch.randn(
                images.size()[0], args.n_z) * args.sigma)
            if torch.cuda.is_available():
                z_fake = z_fake.cuda()

            z_real = encoder(images)

            mmd_loss = imq_kernel(z_real, z_fake, h_dim=encoder.n_z)
            mmd_loss = mmd_loss / batch_size

            with torch.no_grad():
                recon_step_loss += recon_loss.data.item()
                mmd_step_loss += mmd_loss.item()

            total_loss = recon_loss + mmd_loss
            total_loss.backward()

            enc_optim.step()
            dec_optim.step()

            step += 1

            tepoch.set_postfix(
                Reconstruction_Loss=recon_step_loss/step, MMD=mmd_step_loss/step)
            tepoch.update(1)

            current_total_loss = (recon_step_loss + mmd_step_loss) / step

            train_writer_total.add_scalar('loss/mmd',mmd_step_loss/step, epoch)
            train_writer_total.add_scalar('loss/total',current_total_loss, epoch)
            train_writer_total.add_scalar('loss/recon',recon_step_loss/step, epoch)
            
            

    test_recon_step_loss = 0
    test_mmd_step_loss = 0

    with torch.no_grad():

        with tqdm(test_loader) as tepoch:
            test_step = 0

            for (images, _) in test_loader:

                tepoch.set_description(f"Epoch {epoch + 1} TEST")
                if torch.cuda.is_available():
                    images = images.cuda()

                batch_size = images.size()[0]

                z = encoder(images)
                x_recon = decoder(z)

                recon_loss = criterion(x_recon, images)

                z_fake = Variable(torch.randn(
                    images.size()[0], args.n_z) * args.sigma)
                if torch.cuda.is_available():
                    z_fake = z_fake.cuda()

                z_real = encoder(images)

                mmd_loss = imq_kernel(z_real, z_fake, h_dim=encoder.n_z)
                mmd_loss = mmd_loss / batch_size

                test_recon_step_loss += recon_loss.data.item()
                test_mmd_step_loss += mmd_loss.item()

                test_step += 1

                tepoch.set_postfix(
                    Reconstruction_Loss=test_recon_step_loss/test_step, MMD=test_mmd_step_loss/test_step)
                tepoch.update(1)

                test_writer_total.add_scalar('loss/mmd', test_mmd_step_loss/test_step, epoch)
                test_writer_total.add_scalar('loss/total', test_recon_step_loss/test_step + test_mmd_step_loss/test_step, epoch)
                test_writer_total.add_scalar('loss/recon', test_recon_step_loss/test_step, epoch)
                
    # if current_total_loss < last_total_loss:
    #     last_total_loss = current_total_loss
    create_checkpoint(epoch)

    enc_scheduler.step()
    dec_scheduler.step()


train_writer_total.close()
test_writer_total.close()
