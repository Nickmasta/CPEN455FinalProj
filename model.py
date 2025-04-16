import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from layers import *


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=0)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to the left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to the left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=2)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))

        return u, ul


class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=4, nr_filters=100, nr_logistic_mix=10,
                 resnet_nonlinearity='concat_elu', input_channels=3, num_classes=4, embedding_dim=16):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x: concat_elu(x)
        else:
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.num_classes = num_classes
        
        # Early fusion: embed class information to be combined with initial input
        self.class_embedding = nn.Embedding(num_classes, 1)  # Outputs 1 channel for concatenation
        
        # Middle fusion: embeddings to be used in up and down layers
        self.middle_embedding = nn.Embedding(num_classes, embedding_dim)
        self.embed_proj = nn.Conv2d(embedding_dim, nr_filters, kernel_size=1)
        
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                self.resnet_nonlinearity) for i in range(3)])

        self.up_layers = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                    stride=(2, 2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                    nr_filters, stride=(2, 2)) for _ in range(2)])

        self.upsize_u_stream = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                    stride=(2, 2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                    nr_filters, stride=(2, 2)) for _ in range(2)])

        # Modified initial layers to include early fusion (+1 for class channel)
        self.u_init = down_shifted_conv2d(input_channels + 1 + 1, nr_filters, filter_size=(2, 3),
                        shift_output_down=True)

        self.ul_init = nn.ModuleList([
            down_shifted_conv2d(input_channels + 1 + 1, nr_filters,
                               filter_size=(1, 3), shift_output_down=True),
            down_right_shifted_conv2d(input_channels + 1 + 1, nr_filters,
                                    filter_size=(2, 1), shift_output_right=True)
        ])

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

    def forward(self, x, y=None, sample=False):
        # If no y provided (unconditional case), use zeros
        if y is None:
            y = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # similar as done in the tf repo:
        if self.init_padding is None or self.init_padding.shape != x.shape[:2] + (1,) + x.shape[2:]:
            padding = torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device)
            self.init_padding = padding

        if sample:
            padding = torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device)
            x = torch.cat((x, padding), 1)

        ### Prepare class conditioning ###
        # Early fusion: get class channel and expand to spatial dimensions
        y_early = self.class_embedding(y).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        y_early = y_early.expand(-1, -1, x.size(2), x.size(3))  # [B, 1, H, W]
        
        # Middle fusion: prepare embeddings for later use
        y_middle = self.middle_embedding(y)  # [B, embedding_dim]
        y_middle = y_middle.view(y_middle.size(0), y_middle.size(1), 1, 1)  # [B, embedding_dim, 1, 1]
        y_middle = self.embed_proj(y_middle)  # [B, nr_filters, 1, 1]

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
        # Early fusion: concatenate class information with input
        x = torch.cat((x, y_early), dim=1)  # add as additional channel
        
        # Initialize lists
        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        
        for i in range(3):
            # Middle fusion: add class information before each up layer
            u_list[-1] = u_list[-1] + y_middle.expand_as(u_list[-1])
            ul_list[-1] = ul_list[-1] + y_middle.expand_as(ul_list[-1])
            
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list += u_out
            ul_list += ul_out

            if i != 2:
                # downscale (only twice)
                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ###    DOWN PASS    ###
        u = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # Middle fusion: add class information before each down layer
            u = u + y_middle.expand_as(u)
            ul = ul + y_middle.expand_as(ul)
            
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            # upscale (only twice)
            if i != 2:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0

        return x_out


class random_classifier(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(random_classifier, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(3, NUM_CLASSES)
        print("Random classifier initialized")
        # create a folder
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'models')):
            os.makedirs(os.path.join(os.path.dirname(__file__), 'models'))
        torch.save(self.state_dict(), os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth'))
        
    def forward(self, x, device):
        return torch.randint(0, self.NUM_CLASSES, (x.size(0),), device=device)
