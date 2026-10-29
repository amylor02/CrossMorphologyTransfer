import torch
import torch.nn as nn
import pymotion.rotations.ortho6d_torch as ortho6d
from networks.skeleton_layers import (
    SkeletonConv,
    SkeletonUnpool,
    find_neighbor,
)
from networks.unified.components import (
    HeadingConcatAdapter,
    dummy_joint,
    num_layers,
    resolve_vae_latent_dim,
)
from networks.unified.encoder import Encoder


class Decoder(nn.Module):
    def __init__(self, param, enc: Encoder, device, *, variant):
        super(Decoder, self).__init__()
        self.variant = variant

        self.param = param
        self.device = device
        self.layers = nn.ModuleList()
        self.convs = []
        self.is_vae = enc.is_vae
        self.is_vq_vae = enc.is_vq_vae
        self.channels_last_layer = enc.channel_list[-1]
        self.stride = param["stride_encoder_conv"]
        self.vae_latent_dim = resolve_vae_latent_dim(param)
        
        latent_rot_dim = 0
        if(self.is_vae or self.is_vq_vae):
            self.fc_dec = nn.Linear(self.vae_latent_dim + latent_rot_dim, enc.channel_list[-1])
        

        kernel_size = param["kernel_size_temporal_dim"]
        padding = (kernel_size - 1) // 2
        number_layers = num_layers
        default_channels_per_joints = 9
        self.channel_size = [default_channels_per_joints] + [
            default_channels_per_joints * (2**i) for i in range(1, number_layers + 1)
        ]  # first each joint has 8 channels, after collapse 16, then 32...
        for i in range(number_layers):
            seq = []


            if i == 0:
                first_layer_scale = 1
            else:
                first_layer_scale = 1
                
            neighbor_list, _ = find_neighbor(
                enc.parents[number_layers - i - 1], param["neighbor_distance"]
            )
            num_joints = len(neighbor_list)
            self.num_joints = num_joints
            if(dummy_joint):
                num_joints += 1
            unpool = SkeletonUnpool(
                pooling_list=enc.pooling_lists[number_layers - i - 1],
                channels_per_edge=self.channel_size[number_layers - i] * first_layer_scale,
                device=device,
            )
            seq.append(nn.Upsample(scale_factor=self.stride * first_layer_scale, mode="linear", align_corners=False))
            seq.append(unpool)
            seq.append(
                SkeletonConv(
                    param=param,
                    neighbor_list=neighbor_list,
                    kernel_size=kernel_size,
                    in_channels_per_joint=self.channel_size[number_layers - i] * first_layer_scale,
                    out_channels_per_joint=self.channel_size[number_layers - i] // 2,
                    joint_num=num_joints,
                    in_offset_channel=3
                    * enc.channel_size[number_layers - i - 1]
                    // enc.channel_size[0],
                    padding=padding,
                    stride=1,
                    device=device,
                    add_offset=True,
                )
            )
            self.convs.append(seq[-1])
            
            if i != number_layers - 1:
                seq.append(nn.LeakyReLU(negative_slope=0.2))
                #seq.append(ConvLayerNorm(num_joints * self.channel_size[number_layers-i]//2))
            self.layers.append(nn.Sequential(*seq))


        neighbor_list, _ = find_neighbor(
            enc.parents[number_layers - 1], param["neighbor_distance"]
        )
        num_joints_ = len(neighbor_list)

        self.res_upsample_cnn = SkeletonConv(
                param=param,
                neighbor_list=neighbor_list,
                kernel_size=1,
                in_channels_per_joint=self.channel_size[number_layers]*4,
                out_channels_per_joint=self.channel_size[0],
                joint_num=num_joints_,
                in_offset_channel=3 * enc.channel_size[number_layers-1]//enc.channel_size[0],
                padding=0,
                stride=1,
                device=device,
                add_offset=True,
            )
        self.res_upsample = nn.Sequential(
            nn.Upsample(scale_factor=32, mode='linear', align_corners=False),
            SkeletonUnpool(
                pooling_list=enc.pooling_lists[number_layers - 1],
                channels_per_edge=self.channel_size[number_layers]*4,
                device=device
            ),
            self.res_upsample_cnn
        )
        self.fc = nn.Linear(num_joints*8, (num_joints-2)*8)
        

        self.heading_adapter = HeadingConcatAdapter(
            heading_dim=variant.heading_dim,
            latent_channels=self.channels_last_layer,
            stride=self.stride ** number_layers,
            kernel_size=15,
            hidden_dim=max(64, self.channels_last_layer // 4),
        )
        




    def forward(self, input, offset, mean_dqs, std_dqs, denorm_offsets, parents, smooth_root_pos=None,
                root_quats = None, heading=None, ctrl=None):
        
        # if latent_pair is not None:
        #     # (frames, batch_size, latent_dim)
        #     latent = input.permute(2,0,1)
        #     latent_pair = latent_pair.permute(2,0,1)
        #     input = self.cross_attention(latent,latent_pair, latent_pair)
        #     input = input.permute(1,2,0) # (batch_size, latent_dim, frames)

        
        if(root_quats is not None):
            input = torch.cat([input,root_quats.permute(0,2,1)], dim=2) # add root rotations to the input

        input = self.heading_adapter(input, heading)
        


        if(self.is_vae or self.is_vq_vae and False):
            input = self.fc_dec(input) # (batch_size * frames, channels)
            input = input.permute(0, 2, 1)
 
        original = input.clone()
        for i, layer in enumerate(self.layers):
            self.convs[i].set_offset(offset[len(self.layers) - i - 1])   
            input = layer(input)
        
            

        

        
 
        
        
        # input has shape (batch_size, num_joints * 8, frames)
        # denormalize rotations
        # input = torch.cat([root, input[:,9:]],dim=1) #
        # input_joints = input[:,9:] #
        # input_joints = input_joints * std_dqs[9:].unsqueeze(-1) + mean_dqs[9:].unsqueeze(-1) #
        # input = torch.cat([input[:,:9],input_joints],dim=1) #
        
        
        input = input * std_dqs.unsqueeze(-1) + mean_dqs.unsqueeze(-1)
        input = input.reshape(input.shape[0], -1, 9, input.shape[-1]).permute(
            0, 3, 1, 2
        )
        
        
        #input = dquat.normalize(input)
        # input_joints = input[:,:,1:] #
        # input_joints = ortho6d.normalize(input_joints) #
        # input = torch.cat([input[:,:,0:1],input_joints],dim=2) #
        input = ortho6d.normalize(input)
      
        input = input.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        input = (input - mean_dqs.unsqueeze(-1))/ std_dqs.unsqueeze(-1)
        if self.variant.output_clamp is not None:
            input = torch.clamp(input, -self.variant.output_clamp, self.variant.output_clamp)
        return input
