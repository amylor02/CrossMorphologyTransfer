import random
import torch
from torch.utils.data.dataloader import DataLoader
from motion_data import TestMotionData, TrainMotionData, compute_phase
import pymotion.rotations.quat as quat
from pymotion.ops.skeleton import from_root_dual_quat, translation_each_joint
from pymotion.io.bvh import BVH
import numpy as np
from train_data import Train_Data
from generator_architecture import Generator_Model
from ik_architecture import IK_Model
import time
import argparse
import os
import eval_metrics
scale = 1
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

# Train Modes
GENERATOR = 1
IK = 2

human_param = {
    "batch_size": 1024,
    "epochs": 1000,
    "kernel_size_temporal_dim": 7,
    "neighbor_distance": 0,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        4,  # left foot
        8,  # right foot
        11, # chest
        13,  # head
        17,  # left hand
        21,  # right hand
        #22,  #dummy
        
    ],
    "window_size": 64,
    "window_step": 16,
    "seed": 2222,
    "extra_joint": 11 
}

human_param_xsens = {
    "batch_size": 256,
    "epochs": 1000,
    "kernel_size_temporal_dim": 7,
    "neighbor_distance": 2,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        4,  # left foot
        8,  # right foot
        #11, # chest
        13,  # head
        17,  # left hand
        21,  # right hand
        #22,  #dummy
        
    ],
    "window_size": 64,
    "window_step": 16,
    "seed": 2222,
    "extra_joint": -1 #11 
}

#dog params <------- reordered

dog_param_no_tail = {
    "batch_size": 4,
    "epochs": 10000,
    "kernel_size_temporal_dim": 15,
    "neighbor_distance": 2,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        15,  # head
        18,  # left hand
        4,  # right hand
        8,  # left foot
        12,  # right foot
    ],
    "window_size": 64,
    "window_step": 4,
    "seed": 2222,
    "extra_joint": -1,
}



#dog params <-------
dog_param = {
    "batch_size": 128,
    "epochs": 4000,
    "kernel_size_temporal_dim": 7,
    "neighbor_distance": 0,
    "stride_encoder_conv": 3,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        4,  # head
        8,  # left hand
        12,  # right hand
        15,  # left foot
        18,  # right foot
        20,  # tail
    ],
    "window_size": 64,
    "window_step": 8,
    "seed": 2222,
    "extra_joint": -1,
}

cat_param = {
    "batch_size": 16,
    "epochs": 2000,
    "kernel_size_temporal_dim": 15,
    "neighbor_distance": 1,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        5, # tail
        12, #r toe
        18, #l toe
        27, #r finger
        33, #l finger
        43, # head
    ],
    "window_size": 64,
    "window_step": 16,
    "seed": 2222,
    "extra_joint": -1,
}

param = dog_param
rm_flag= False
frame_step = 1

assert param["kernel_size_temporal_dim"] % 2 == 1


def main(args):
    torch.manual_seed(param["seed"])
    random.seed(param["seed"])
    np.random.seed(param["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    train_eval_dir = args.data_path
    train_dir = os.path.join(train_eval_dir, "train")
    if not os.path.exists(train_dir):
        raise ValueError("train directory does not exist")
    train_files = os.listdir(train_dir)
    eval_dir = os.path.join(train_eval_dir, "eval")
    if not os.path.exists(eval_dir):
        raise ValueError("eval directory does not exist")
    eval_files = os.listdir(eval_dir)
    train_dataset = TrainMotionData(param, scale, device)
    eval_dataset = TestMotionData(param, scale, device)
    reference_parents = None  # used to make sure all bvh have the same structure

    rots_missing_frames_list = []
    ground_truth_rots_list = []
    phase_list = []
    velocities_list = []
    phase_chunks = []
    velocity_chunks = []
    for filename in train_files:
        if filename[-4:] == ".bvh":
            
            bvh_from_disk = get_bvh_from_disk(train_dir, filename)
            rots, pos, parents, offsets, _, rots_missing_frames = get_info_from_bvh(
                bvh_from_disk, get_missing_frames= True
            )
            #rots[0,0,:] = rots_missing_frames[0,0,:]
            ########################
            # create chunks of 256 frames each (for root predicotr)
            # list_og_rots = rots_missing_frames[0]
            # list_missing = rots_missing_frames[1]
            # num_frames = list_missing[0].shape[0]
            # chunk_size = 256
            # for i in range(frame_step):
            #     rots_missing_frames = list_missing[i]
            #     true_rots = list_og_rots[i]
            #     for start in range(0, num_frames, chunk_size):
            #         end = start + chunk_size
            #         current_chunk = torch.tensor(rots_missing_frames[start:end]).float()
                    
            #         # Check if current chunk is smaller than chunk_size and pad if necessary
            #         if current_chunk.shape[0] < chunk_size:
            #             padding = torch.zeros((chunk_size - current_chunk.shape[0], current_chunk.shape[1]))
            #             padding_gr_tr = torch.zeros((chunk_size - current_chunk.shape[0], ground_truth_chunk.shape[1]))
            #             current_chunk = torch.cat((current_chunk, padding), dim=0)
            #             ground_truth_chunk = torch.tensor(true_rots[start:end, 0, :]).float()  # actual root rotations
            #             ground_truth_chunk = torch.cat((ground_truth_chunk, padding_gr_tr.clone()), dim=0)
            #         else:
            #             ground_truth_chunk = torch.tensor(true_rots[start:end, 0, :]).float()

            #         rots_missing_frames_list.append(current_chunk)
            #         ground_truth_rots_list.append(ground_truth_chunk)
            #########################
            
            if reference_parents is None:
                reference_parents = parents.copy()
            assert (
                reference_parents == parents
            )  # make sure all bvh have the same structure
            # Train Dataset
            pos_all_joints = translation_each_joint(rots, pos[:,0,:], parents, offsets)
            train_dataset.add_motion(
                offsets,
                pos[:, 0, :],  # only global position
                rots,
                parents,
                pos_all_joints
            )
            chunk_size = 512
            num_frames = pos_all_joints.shape[0]

            # Initialize lists to store results

            # Process pos_all_joints in chunks of 512 frames
            for start_index in range(0, num_frames, chunk_size):
                end_index = min(start_index + chunk_size, num_frames)  # Ensure we do not exceed array bounds
                pos_chunk = pos_all_joints[start_index:end_index]  # Shape: [frames_in_chunk, joints, 3]
                
                # Compute phase and velocity for the current chunk
                phase, velocity_xz = compute_phase(pos_chunk, indices_only=False)
                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(12, 6))
                # # Plot Part 1
                # plt.subplot(2, 1, 1)
                # plt.plot(phase[0], label='Real Phase Part 1', color='blue')
                # plt.legend()
                
                # plt.tight_layout()
                # plt.show()

                # Pad phase and velocity if necessary
                if phase.shape[1] < chunk_size:
                    phase = np.pad(phase, ((0, 0), (0, chunk_size - phase.shape[1])), 'constant')  # Pad along frames dimension
                if len(velocity_xz) < chunk_size:
                    velocity_xz = np.pad(velocity_xz, (0, chunk_size - len(velocity_xz)), 'constant')  # Pad with zeros

                phase_chunks.append(phase)  # Shape: [2, frames]
                velocity_chunks.append(velocity_xz.reshape(1, -1))  # Shape: [1, frames]


    print(len(rots_missing_frames_list), len(ground_truth_rots_list))

    # Once all train files are added, compute the means and stds and normalize
    train_dataset.normalize()
    eval_dataset.set_means_stds(train_dataset.means, train_dataset.stds)
    # Eval Files
    for filename in eval_files:
        if filename[-4:] == ".bvh":
            rots, pos, parents, offsets, bvh = get_info_from_bvh(
                get_bvh_from_disk(eval_dir, filename)
            )
            assert (
                reference_parents == parents
            )  # make sure all bvh have the same structure
            pos_all_joints = translation_each_joint(rots, pos[:,0,:], parents, offsets)
            # Eval Dataset
            eval_dataset.add_motion(
                offsets,
                pos[:, 0, :],  # only global position
                rots,
                parents,
                bvh,
                filename,
                pos_all_joints,
            )
    # Once all eval files are added, normalize
    eval_dataset.normalize()

    train_dataloader = DataLoader(train_dataset, param["batch_size"], shuffle=True)
    train_data = Train_Data(device, param)
    generator_model = Generator_Model(device, param, reference_parents, train_data, is_vae = True).to(
        device
    )
    if args.train_mode & IK != 0:
        ik_model = IK_Model(device, param, reference_parents, train_data).to(device)
    train_data.set_means(train_dataset.means["dqs"])
    train_data.set_stds(train_dataset.stds["dqs"])
    print(args)
    print(args.train_mode & IK)
    _, generator_path, ik_path = get_model_paths(args.name, train_eval_dir)
    #if args.train_mode & GENERATOR == 0 or (args.load and args.train_mode & IK != 0):
    if args.train_mode & GENERATOR == 0 or (args.load):
        print("loading pretrained model")
        # Generator is always needed with IK, load it if not training it
        load_model(generator_model, generator_path, train_data, device)
    if args.train_mode & IK != 0 and args.load:
        load_model(ik_model, ik_path, train_data, device)

    if (args.train_mode & GENERATOR == 0 or args.train_mode & IK == 0) and args.load:
        # Check previous best evaluation loss
        results = evaluate_generator(generator_model, train_data, eval_dataset)
        if args.train_mode & IK != 0:
            results_ik = evaluate_ik(ik_model, results, train_data, eval_dataset)
            results = results_ik
        mpjpe, mpeepe = eval_save_result(
            results,
            train_dataset.means,
            train_dataset.stds,
            eval_dir,
            device,
            save=False,
        )
        best_evaluation = mpjpe + mpeepe
    else:
        best_evaluation = float("inf")
    # Training Loop
        start_time = time.time()
    root_loss_only = True
    for epoch in range(param["epochs"]):
        avg_train_loss = 0.0
        avg_kld_loss = 0.0
        avg_kld_loss_s = 0.0
        avg_kld_loss_root = 0.0
        for step, (denorm_motion, norm_motion) in enumerate(train_dataloader):
            train_data.set_offsets(norm_motion["offsets"], denorm_motion["offsets"])
            train_data.set_motions(
                norm_motion["dqs"],
                norm_motion["displacement"],
            )
            train_data.set_phase(denorm_motion["phase"])
            if args.train_mode & GENERATOR != 0:
                generator_model.train()
            if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
                res_decoder = generator_model.forward()
            if args.train_mode & IK != 0:
                ik_model.train()
                ik_model.forward(res_decoder)
            loss = 0.0
            kld_loss = 0.0
            if args.train_mode & GENERATOR != 0:

                loss_generator, kld, kld_s, kld_r, mu_s,var_s, mu_shat, var_shat, mu_z, var_z = generator_model.optimize_parameters_vae(param["batch_size"],epoch,root_loss_only)
                mu_s = torch.mean(mu_s).item()
                var_s = torch.mean(var_s).item()
                mu_shat = torch.mean(mu_shat).item()
                var_shat = torch.mean(var_shat).item()
                mu_z = torch.mean(mu_z)
                var_z = torch.mean(var_z)

                loss = loss_generator.item()
                kld_loss = kld.item()
                kld_loss_s = kld_s.item()
                kld_loss_r = kld_r.item()

            if args.train_mode & IK != 0:
                loss_ik = ik_model.optimize_parameters(res_decoder)
                loss += loss_ik.item()
                
            avg_train_loss += loss
            avg_kld_loss += kld_loss
            avg_kld_loss_s += kld_loss_s
            avg_kld_loss_root += kld_loss_r
            # Evaluate & Print
            if step == len(train_dataloader) - 1:
                if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
                    results = evaluate_generator(
                        generator_model, train_data, eval_dataset
                    )
                    if args.train_mode & IK != 0:
                        results_ik = evaluate_ik(
                            ik_model,
                            results,
                            train_data,
                            eval_dataset,
                        )
                        results = results_ik
                    if(epoch% 50 == 0 ):
                        save = True
                    else:
                        save = False
                    mpjpe, mpeepe = eval_save_result(
                        results,
                        train_dataset.means,
                        train_dataset.stds,
                        eval_dir,
                        device,
                        save=save,
                    )
                    evaluation_loss = mpjpe + mpeepe
                was_best = False
                if evaluation_loss < best_evaluation:
                    save_model(
                        generator_model if args.train_mode & GENERATOR != 0 else None,
                        ik_model if args.train_mode & IK != 0 else None,
                        train_dataset,
                        args.name,
                        train_eval_dir,
                    )
                    best_evaluation = evaluation_loss
                    was_best = True
                    
                elif epoch % 10 == 0 and epoch!= 0: 
                    save_model_shared(
                        generator_model if args.train_mode & GENERATOR != 0 else None,
                        ik_model if args.train_mode & IK != 0 else None,
                        train_dataset,
                        args.name,
                        train_eval_dir,
                    )

                # Print
                avg_train_loss /= len(train_dataloader)
                avg_kld_loss /= len(train_dataloader)
                avg_kld_loss_s /= len(train_dataloader)
                avg_kld_loss_root /= len(train_dataloader)
                if(avg_kld_loss_root > 0.102):
                    root_loss_only = True
                else:
                    root_loss_only = False

                if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
                    # print(
                    #     "Epoch: {} - Train Loss: {:.4f} - KLD: {:.4f} - KLD_S: {:.4f} - mu,var: {:.4f},{:.4f} - Eval Loss: {:.4f} - MPJPE: {:.4f} - MPEEPE: {:.4f}".format(
                    #         epoch, avg_train_loss, avg_kld_loss, avg_kld_loss_s,mu_s,var_s,evaluation_loss, mpjpe, mpeepe
                    #     )
                    #         + ("*" if was_best else "")
                    # )
                                        print(
                        "Epoch: {} - Train Loss: {:.4f} - KLD: {:.4f} - KLD_S: {:.4f} - KLD_R: {:.8f} - z: {:.4f},{:.4f}  - s: {:.4f},{:.4f} - s_hat: {:.4f},{:.4f} - Eval Loss: {:.4f}".format(
                            epoch, avg_train_loss, avg_kld_loss, avg_kld_loss_s,avg_kld_loss_root,mu_z,var_z,mu_s,var_s,mu_shat,var_shat,evaluation_loss,
                        )
                            + ("*" if was_best else "")
                    )

    end_time = time.time()
    print("Training Time:", end_time - start_time)

    if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
        load_model(generator_model, generator_path, train_data, device)
        results = evaluate_generator(generator_model, train_data, eval_dataset)
        if args.train_mode & IK != 0:
            load_model(ik_model, ik_path, train_data, device)
            results_ik = evaluate_ik(ik_model, results, train_data, eval_dataset)
            results = results_ik

        mpjpe, mpeepe = eval_save_result(
            results, train_dataset.means, train_dataset.stds, eval_dir, device
        )
        evaluation_loss = mpjpe + mpeepe

    print("Evaluate Loss: {}".format(evaluation_loss))
    if args.train_mode & (GENERATOR | IK) != 0:
        print("Mean Per Joint Position Error: {}".format(mpjpe))
        print("Mean End Effector Position Error: {}".format(mpeepe))


def eval_save_result(results, train_means, train_stds, eval_dir, device, save=True):
    array_mpjpe = np.empty((len(results),))
    array_mpeepe = np.empty((len(results),))
    for step, (res, bvh, filename) in enumerate(results):
        if save:
            eval_path, eval_filename = result_to_bvh(
                res, train_means, train_stds, bvh, filename
            )
            # Evaluate Positional Error
            mpjpe, mpeepe = eval_metrics.eval_pos_error(
                get_bvh_from_disk(eval_dir, filename),
                get_bvh_from_disk(eval_path, eval_filename, True),#<---------
                device,
                param["sparse_joints"][1:],
            )
        else:
            result_to_bvh(res, train_means, train_stds, bvh, None, save=False)
            # Evaluate Positional Error
            mpjpe, mpeepe = eval_metrics.eval_pos_error(
                get_bvh_from_disk(eval_dir, filename),
                bvh,
                device,
                param["sparse_joints"][1:],
            )

        array_mpjpe[step] = mpjpe
        array_mpeepe[step] = mpeepe

    return np.mean(array_mpjpe), np.mean(array_mpeepe)


# def load_model(model, model_path, train_data, device):
#     model_name = os.path.basename(model_path)[: -len(".pt")]
#     assert model_name == "generator" or model_name == "ik"
#     if model_name == "generator":
#         data_path = model_path[: -len("generator.pt")] + "data.pt"
#         checkpoint = torch.load(model_path, map_location=device)
#         model.load_state_dict(checkpoint["model_state_dict"])
#     elif model_name == "ik":
#         data_path = model_path[: -len("ik.pt")] + "data.pt"
#         checkpoint = torch.load(model_path, map_location=device)
#         model.load_state_dict(checkpoint["model_state_dict"])
#     data = torch.load(data_path, map_location=device)
#     means = data["means"]
#     stds = data["stds"]
#     train_data.set_means(means["dqs"])
#     train_data.set_stds(stds["dqs"])
#     return means, stds

def load_model(model, model_path, train_data, device, ignore_transform_net = True):
    model_name = os.path.basename(model_path)[: -len(".pt")]
    assert model_name == "generator" or model_name == "ik"
    if model_name == "generator":
        data_path = model_path[: -len("generator.pt")] + "data.pt"
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        
        # dont load transform net
        #if ignore_transform_net:
        #    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.transform_net.')}
        
        #state_dict = {k: v for k, v in state_dict.items() if (not 'lstm' in k)}
        model.load_state_dict(state_dict, strict=False)
        
    elif model_name == "ik":
        data_path = model_path[: -len("ik.pt")] + "data.pt"
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    data = torch.load(data_path, map_location=device)
    means = data["means"]
    stds = data["stds"]
    train_data.set_means(means["dqs"])
    train_data.set_stds(stds["dqs"])
    return means, stds

def get_model_paths(name, train_eval_dir):
    model_name = (
        "model_" + name + "_" + os.path.basename(os.path.normpath(train_eval_dir))
    )
    model_dir = os.path.join("models", model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    data_path = os.path.join(model_dir, "data.pt")
    generator_path = os.path.join(model_dir, "generator.pt")
    ik_path = os.path.join(model_dir, "ik.pt")
    return data_path, generator_path, ik_path

def get_model_paths_shared(name, train_eval_dir):
    model_name = (
        "model_" + name + "_" + os.path.basename(os.path.normpath(train_eval_dir))
    )
    model_dir = os.path.join("models", model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    data_path = os.path.join(model_dir, "best_root/data.pt")
    generator_path = os.path.join(model_dir, "best_root/generator.pt")
    ik_path = os.path.join(model_dir, "ik.pt")
    return data_path, generator_path, ik_path


def save_model(
    generator_model,
    ik_model,
    train_dataset,
    name,
    train_eval_dir,
):
    data_path, generator_path, ik_path = get_model_paths(name, train_eval_dir)

    if train_dataset is not None:
        torch.save(
            {
                "means": train_dataset.means,
                "stds": train_dataset.stds,
            },
            data_path,
        )
    if generator_model is not None:
        torch.save(
            {
                "model_state_dict": generator_model.state_dict(),
            },
            generator_path,
        )
    if ik_model is not None:
        torch.save(
            {
                "model_state_dict": ik_model.state_dict(),
            },
            ik_path,
        )

def save_model_shared(
    generator_model,
    ik_model,
    train_dataset,
    name,
    train_eval_dir,
):
    data_path, generator_path, ik_path = get_model_paths_shared(name, train_eval_dir)
    #generator_path = os.path.join(generator_path,"best_root")

    if train_dataset is not None:
        torch.save(
            {
                "means": train_dataset.means,
                "stds": train_dataset.stds,
            },
            data_path,
        )
    if generator_model is not None:
        torch.save(
            {
                "model_state_dict": generator_model.state_dict(),
            },
            generator_path,
        )
    if ik_model is not None:
        torch.save(
            {
                "model_state_dict": ik_model.state_dict(),
            },
            ik_path,
        )



def get_bvh_from_disk(path, filename, remove=True):
    remove=rm_flag
    path = os.path.join(path, filename)
    bvh = BVH()
    bvh.load(path)
    #remove lower body joints
    if remove:
        bvh.remove_joints([19,20])
    return bvh


def get_info_from_bvh(bvh, incremental_rots = True, get_missing_frames = False, get_phase = False):
    rot_roder = np.tile(bvh.data["rot_order"][0], (bvh.data["rotations"].shape[0], bvh.data["rotations"].shape[1], 1))  # made a change here
    rots = quat.unroll(
        quat.from_euler(np.radians(bvh.data["rotations"]), order=rot_roder),
        axis=0,
    )
    rots = quat.normalize(rots)  # make sure all quaternions are unit quaternions
    og_rots = rots.copy()
    
    # rots = quat.normalize(rots)
    pos = bvh.data["positions"]
    parents = bvh.data["parents"]
    parents[0] = 0  # BVH sets root as None
    offsets = bvh.data["offsets"]
    offsets[0] = np.zeros(3)  # force to zero offset for root joint
    
    if(incremental_rots):
        rots = quat.compute_incremental_quaternions(og_rots)
        #rots = quat.compute_incremental_intermittent_quaternions(rots)
        
    if(get_missing_frames):
        list_og_rots = []
        list_missing_frames = []
        foot_1_idx=15
        foot_2_idx=18
        #og_rots= quat.compute_incremental_quaternions(og_rots)
        
        for i in range(frame_step):            
            #rots_missing_frames = quat.compute_filtered_quaternions(og_rots[i:,:,:],pos[i:,0,:])#
            rots_missing_frames = quat.compute_incremental_intermittent_quaternions(og_rots[i:,:,:], pos[i:,0,:])
            phase, _  = compute_phase(pos[i:],joint_id=[0])
            phase = phase.squeeze()
            rots_missing_frames = np.concatenate([rots_missing_frames,phase[:,np.newaxis]],axis=1)
            #rots_missing_frames =  rots_filtered_frames
            #rots_missing_frames = quat.compute_incremental_intermittent_quaternions(og_rots[i:,:,:], pos[i:,0,:])
            #rots_missing_frames[:,4:] = rots_filtered_frames[:,4:].copy()
            og_quats = og_rots[i:,:,:]#og_quats = quat.compute_incremental_quaternions(og_rots[i:,:,:])
            displacement = pos[i+1:, 0, :] - pos[i:-1, 0, :]
            displacement = np.vstack((np.zeros((1, 3)), displacement))
            true_values = np.zeros((og_quats.shape[0], og_quats.shape[1], 7)) # [frames, joints, quat(4)+displacement(3)+4+4(feet)]
            #true_values[:,0,:] = np.concatenate((og_quats[:,0,:], og_quats[:,foot_1_idx,:], og_quats[:,foot_2_idx,:], displacement), axis=-1) # quaternion + displacement [w,x,y,z] + [x,y,z]
            true_values[:,0,:] = np.concatenate((og_quats[:,0,:], displacement), axis=-1)
            list_og_rots.append(true_values)    # actual rotations and displacements
            list_missing_frames.append(rots_missing_frames)
        
        return rots, pos, parents, offsets, bvh, (list_og_rots, list_missing_frames)
    
    if(get_phase):
        list_og_rots = []
        list_missing_frames = []
        foot_1_idx=15
        foot_2_idx=18
        #og_rots= quat.compute_incremental_quaternions(og_rots)
        
      
        
        return rots, pos, parents, offsets, bvh, (list_og_rots, list_missing_frames)

    return rots, pos, parents, offsets, bvh


def evaluate_generator(generator_model, train_data, dataset, sparse_motions=None):
    # WARNING: means and stds for the model are not set in this function... they should be set before
    generator_model.eval()
    results = []
    with torch.no_grad():
        for index in range(dataset.get_len()):
            norm_motion = dataset.get_item(index)
            train_data.set_offsets(
                norm_motion["offsets"].unsqueeze(0),
                norm_motion["denorm_offsets"].unsqueeze(0),
            )
            train_data.set_motions(
                norm_motion["dqs"].unsqueeze(0),
                norm_motion["displacement"].unsqueeze(0),
            )
            train_data.set_phase(
                torch.tensor(norm_motion["phase"], dtype=torch.float32).to('cuda').unsqueeze(0)
            )
            if sparse_motions is not None:
                train_data.set_sparse_motion(sparse_motions[index])
            res = generator_model.forward()
            bvh, filename = dataset.get_bvh(index)
            results.append((res, bvh, filename))
    return results


def evaluate_ik(ik_model, results_decoder, train_data, dataset):
    # WARNING: means and stds for the model are not set in this function... they should be set before
    ik_model.eval()
    results = []
    with torch.no_grad():
        for index in range(dataset.get_len()):
            norm_motion = dataset.get_item(index)
            train_data.set_offsets(
                norm_motion["offsets"].unsqueeze(0),
                norm_motion["denorm_offsets"].unsqueeze(0),
            )
            train_data.set_motions(
                norm_motion["dqs"].unsqueeze(0),
                norm_motion["displacement"].unsqueeze(0),
            )
            res = ik_model.forward(results_decoder[index][0])
            bvh, filename = dataset.get_bvh(index)
            results.append((res, bvh, filename))
    return results


def run_set_data(train_data, dataset):
    with torch.no_grad():
        norm_motion = dataset.get_item()
        train_data.set_offsets(
            norm_motion["offsets"].unsqueeze(0),
            norm_motion["denorm_offsets"].unsqueeze(0),
        )
        train_data.set_motions(
            norm_motion["dqs"].unsqueeze(0),
            norm_motion["displacement"].unsqueeze(0),
        )


def run_generator(model):
    # WARNING: means and stds for the model are not set in this function... they should be set before
    model.eval()
    with torch.no_grad():
        res_decoder = model.forward()
    return res_decoder


def run_ik(model, res_decoder, frame=None):
    # WARNING: means and stds for the model are not set in this function... they should be set before
    model.eval()
    with torch.no_grad():
        res = model.forward(res_decoder, frame)
    return res


def result_to_bvh(res, means, stds, bvh, filename, save=True, initial_frame = None, feet_idx = None):
    res = res.permute(0, 2, 1)
    res = res.flatten(0, 1)
    res = res.cpu().detach().numpy()
    # get dqs and displacement
    dqs = res
    # denormalize
    dqs = dqs * stds["dqs"].cpu().numpy() + means["dqs"].cpu().numpy()
    # get rotations and translations from dual quatenions
    dqs = dqs.reshape(dqs.shape[0], -1, 8)
    trans, rots = from_root_dual_quat(dqs, np.array(bvh.data["parents"]))
    #zeros = np.zeros((pos.shape[0], pos.shape[1], 1))
    # pos = np.concatenate([pos, zeros], axis=-1) # because displacement has 4 items (x,y,z increments and actual y)
    # pos = pos * stds["displacement"].cpu().numpy() + means["displacement"].cpu().numpy()
    # pos[1:] = pos[1:] + pos[:-1]
    #phase = compute_phase(pos)

    rot_roder = np.tile(bvh.data["rot_order"][0], (rots.shape[0], rots.shape[1], 1))
    
    if(initial_frame is not None):
        rots[:,0,:] = initial_frame[:rots.shape[0],:]
        #rots[0,0,:] = initial_frame
        #rots[:,0,:] = initial_frame[:rots.shape[0], :]
    #initial_rotation = quat.from_euler(bvh.data["rotations"][0,0,:], rot_roder[0,0,:])
    #rots[0,0,:] = initial_rotation
    rots = quat.compute_cumulative_quaternions(rots, feet_idx)
    # rots[:,0,:] = initial_frame[:rots.shape[0], :]
    # quaternions to euler

    #rotations = np.degrees(quat.to_euler(rots, order=rot_roder))
    rotations = bvh.to_degrees(quat.to_euler(rots, order=rot_roder))
    bvh.data["rotations"] = rotations
    # positions
    positions = bvh.data["positions"][: rotations.shape[0]]
    # from scipy.ndimage import gaussian_filter1d
    # for i in range (positions.shape[2]):
    #     positions[:,0,i] = gaussian_filter1d(positions[:,0,i],sigma=2)
    bvh.data["positions"] = positions
    # save
    bvh.data["parents"][0] = None  # BVH sets root as None
    path = None
    
    # import matplotlib.pyplot as plt
    # pos_real = bvh.data["positions"][: rotations.shape[0]]
    # phase_real = compute_phase(pos_real).squeeze()
    # # Convert the tensor to CPU and NumPy for plotting
    # phase_data = phase.squeeze()#.cpu().numpy()

    # # Create a new figure
    # plt.figure(figsize=(10, 6))

    # # Plot the phase data
    # plt.plot(phase_data, marker='o', linestyle='-')
    # plt.plot(phase_real, marker='x', linestyle='-')

    # # Add labels and title
    # plt.xlabel('Index')
    # plt.ylabel('Phase Value')
    # plt.title('Phase Values Over Time')

    # # Show the plot
    # plt.show()
    
    
    if save:
        path = "data"
        filename = "eval_" + filename
        bvh.save(os.path.join(path, filename))
    return path, filename

# from scipy.signal import find_peaks
# from scipy.ndimage import gaussian_filter1d

# def compute_phase(pos):
#     # Assuming x_values, z_values, and y_values are 1D arrays representing the movement over time
#     x_values = pos[:,0,0]  # X-axis movement data
#     z_values = pos[:,0,2]  # Z-axis movement data
#     y_values = pos[:,0,1]  # Y-axis movement data
#     epsilon = 0.01  # Stationary threshold

#     # Step 2: Compute xz plane velocity
#     vx = np.diff(x_values, prepend=x_values[0])
#     vz = np.diff(z_values, prepend=z_values[0])
#     velocity_xz = np.sqrt(vx**2 + vz**2)

#     # Step 3: Discard low velocities
#     velocity_threshold = 0.01  # Set your threshold here
#     active_indices = np.where(velocity_xz > velocity_threshold)[0]
#     inactive_indices = np.where(velocity_xz < velocity_threshold)[0]
#     # Step 4: Extract active y values
#     active_y_values = y_values[active_indices]

#     # Step 5: Smooth the active y values for better peak detection
#     smoothed_y = gaussian_filter1d(y_values, sigma=5)
    
#     # Step 7: Locate peaks and valleys
#     peaks, _ = find_peaks(smoothed_y,)
#     valleys, _ = find_peaks(-smoothed_y,)

#     starting_point = min(peaks[0],valleys[0])
    
#     def construct_signal(peaks, valleys, num_points):
#         """
#         """
#         # Initialize the signal array
#         signal = np.zeros(num_points)
#         # Ensure peaks and valleys are sorted
#         points = np.concatenate((peaks,valleys))
#         points = sorted(points)
#         # Iterate through the points to construct the signal
#         for i in range(len(points) - 1):
#             start = points[i]
#             end = points[i + 1]
#             x = np.linspace(0, 1, end - start)  # Normalize to 0 to 1
#             if points[i] in peaks:
#                 # If current point is a peak
#                 signal[start:end] = 0.5 * (1 + np.cos(np.pi * x))  # Descend to valley (sine wave)
#             elif points[i] in valleys:
#                 # If current point is a valley
#                 signal[start:end] = 0.5 * (1 - np.cos(np.pi * x))  # Ascend to peak (sine wave)
        
#         return signal

    
#     num_points = len(smoothed_y)
#     phase = construct_signal(peaks, valleys, num_points)
#     return phase
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Motion Upsampling Network")
    parser.add_argument(
        "data_path",
        type=str,
        help="path to data directory containing one or multiple .bvh for training, last .bvh is used as test data",
    )
    parser.add_argument(
        "name",
        type=str,
        help="name of the experiment, used to save the model and the logs",
    )
    parser.add_argument(
        "train_mode",
        type=str.lower,
        choices=["generator", "ik", "all"],
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="load the model(s) from a checkpoint",
    )
    args = parser.parse_args()
    if args.train_mode == "generator":
        args.train_mode = GENERATOR
    elif args.train_mode == "ik":
        args.train_mode = IK
    elif args.train_mode == "all":
        args.train_mode = GENERATOR | IK
    main(args)
