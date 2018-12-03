import torch.nn as nn



class DCGenerator(nn.Module):


    def __init__(self, image_size, nc):
        ngf = 64
        nz = 100
        super(DCGenerator, self).__init__()
        #self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # PrintingModule(11),
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            # PrintingModule(12),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            # PrintingModule(13),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            # PrintingModule(14),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            # PrintingModule(15),
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            # PrintingModule(16),
            nn.ConvTranspose2d( ngf, nc, 1, 1, 0, bias=False),
            # PrintingModule(17),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, data):
        # TODO your code here
        return self.main(data)
        

class PrintingModule(nn.Module):
    def __init__(self, number):
        super(PrintingModule, self).__init__()
        self.number = number
    def forward(self, data):
        print("i am pritner number " + str(self.number) + ", shape of my input is " + str(data.shape))
        return data

class DCDiscriminator(nn.Module):


    def __init__(self, image_size, nc):
        ndf = 64
        super(DCDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x    // 3 * 32 * 32
            # PrintingModule(1),
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32 // 64 * 16 * 16
            # PrintingModule(2),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            # PrintingModule(3),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            # PrintingModule(4),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # PrintingModule(5),
            nn.Conv2d(ndf * 8, 1, image_size // 16, 1, 0, bias=False),
            # PrintingModule(6),
            nn.Sigmoid()
        )

    def forward(self, data):
        return self.main(data)
        # TODO your code here

__all__ = ['DCGenerator', 'DCDiscriminator']
