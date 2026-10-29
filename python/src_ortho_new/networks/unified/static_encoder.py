import torch.nn as nn
from networks.skeleton_layers import (
    SkeletonLinear,
    SkeletonPool,
    find_neighbor,
)
from networks.unified.components import (
    dummy_joint,
    num_layers,
)


# encoder for static part, i.e. offset part
class StaticEncoder(nn.Module):
    def __init__(self, param, parents, device):
        super(StaticEncoder, self).__init__()
        self.layers = nn.ModuleList()
        channels = 3  # position

        number_layers = num_layers
        layer_parents = parents
        
        for i in range(number_layers):
            neighbor_list, _ = find_neighbor(layer_parents, param["neighbor_distance"])
          
            seq = []
            if(not dummy_joint):
                seq.append(
                    SkeletonLinear(
                        param=param,
                        neighbor_list=neighbor_list,
                        in_channels=channels * len(neighbor_list),
                        out_channels=channels * 2 * len(neighbor_list),
                        device=device,
                    )
                )
            else:
                seq.append(
                    SkeletonLinear(
                        param=param,
                        neighbor_list=neighbor_list,
                        in_channels=channels * (len(neighbor_list)+1),
                        out_channels=channels * 2 * (len(neighbor_list)+1),
                        device=device,
                    )
                )


            if i < number_layers - 1:
                pool = SkeletonPool(
                    parents=layer_parents, channels_per_edge=channels * 2, device=device
                )
                layer_parents = pool.new_parents
                seq.append(pool)
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            channels *= 2
            self.layers.append(nn.Sequential(*seq))

    def forward(self, input):
        input = input.reshape(input.shape[0], -1)
        output = [input]
  
        for layer in self.layers:
            input = layer(input)
            input = input.squeeze(-1)
            output.append(input)
        return output
