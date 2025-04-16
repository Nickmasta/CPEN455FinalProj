import torch.nn as nn
from layers import *


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=0)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list  += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=2)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))

        return u, ul


class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=4, nr_filters=100, nr_logistic_mix=10,
                resnet_nonlinearity='concat_elu', input_channels=3, 
                num_classes=4, embedding_dim=16):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x: concat_elu(x)
        else:
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Early fusion modifications
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        # Project embedding to match spatial dimensions
        self.embed_proj = nn.Conv2d(embedding_dim, nr_filters, kernel_size=1)
        
        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                        self.resnet_nonlinearity) for i in range(3)])

        self.up_layers = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                        self.resnet_nonlinearity) for _ in range(3)])

        # Modify initial convolutions to account for embedding
        self.u_init = down_shifted_conv2d(input_channels + 1 + nr_filters, nr_filters, 
                                        filter_size=(2,3), shift_output_down=True)

        self.ul_init = nn.ModuleList([
            down_shifted_conv2d(input_channels + 1 + nr_filters, nr_filters,
                              filter_size=(1,3), shift_output_down=True),
            down_right_shifted_conv2d(input_channels + 1 + nr_filters, nr_filters,
                                   filter_size=(2,1), shift_output_right=True)
        ])

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

    def forward(self, x, class_labels=None, sample=False):
        if self.init_padding is None or self.init_padding.shape[0] != x.shape[0]:
            padding = torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device)
            self.init_padding = padding

        if sample:
            padding = torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device)
            x = torch.cat((x, padding), 1)

        ### Early Fusion ###
        if class_labels is not None:
            # Get class embeddings
            h_class = self.class_embedding(class_labels)  # [B, embedding_dim]
            # Project to nr_filters and expand spatially
            h_class = self.embed_proj(h_class.unsqueeze(-1).unsqueeze(-1))  # [B, nr_filters, 1, 1]
            h_class = h_class.expand(-1, -1, x.size(2), x.size(3))  # [B, nr_filters, H, W]
        else:
            # If no class provided, use zeros
            h_class = torch.zeros(x.size(0), self.nr_filters, x.size(2), x.size(3), device=x.device)

        # Prepare input with padding and class information
        x = x if sample else torch.cat((x, self.init_padding), 1)
        x = torch.cat((x, h_class), dim=1)  # Concatenate along channel dimension

        ### UP PASS ###
        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        
        for i in range(3):
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list += u_out
            ul_list += ul_out

            if i != 2:
                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ### DOWN PASS ###
        u = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            if i != 2:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))
        return x_out
    
    
class random_classifier(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(random_classifier, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(3, NUM_CLASSES)
        print("Random classifier initialized")
        # create a folder
        if os.path.join(os.path.dirname(__file__), 'models') not in os.listdir():
            os.mkdir(os.path.join(os.path.dirname(__file__), 'models'))
        torch.save(self.state_dict(), os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth'))
    def forward(self, x, device):
        return torch.randint(0, self.NUM_CLASSES, (x.shape[0],)).to(device)
    
    
