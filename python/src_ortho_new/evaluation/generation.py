import torch


def evaluate_generator(generator_model, train_data, dataset, sparse_motions=None):
    # WARNING: means and stds for the model are not set in this function... they should be set before
    generator_model.eval()
    results = []
    yaw_diffs = []
    disps = []
    with torch.no_grad():
        for index in range(dataset.get_len()):
            norm_motion = dataset.get_item(index)
            train_data.set_offsets(
                norm_motion["offsets"].unsqueeze(0),
                norm_motion["denorm_offsets"].unsqueeze(0),
            )
            train_data.set_end_sites(
                torch.tensor(norm_motion["end_sites"], dtype=torch.float32)
                .to("cuda")
                .unsqueeze(0),
                torch.tensor(norm_motion["end_sites_parents"], dtype=torch.float32)
                .to("cuda")
                .unsqueeze(0),
            )
            train_data.set_motions(
                norm_motion["dqs"].unsqueeze(0),
                norm_motion["displacement"].unsqueeze(0),
                norm_motion["disp_8"].unsqueeze(0),
                norm_motion["tags"]["sin_diff"].unsqueeze(0),
                norm_motion["tags"]["cos_diff"].unsqueeze(0),
                norm_motion["loss_weights"].unsqueeze(0),
            )
            train_data.set_rots(
                norm_motion["rots"].unsqueeze(0),
            )
            train_data.set_phase(
                torch.tensor(norm_motion["phase"], dtype=torch.float32)
                .to("cuda")
                .unsqueeze(0)
            )
            train_data.set_phase_per_8_frames(
                torch.tensor(norm_motion["phase_per_8_frames"], dtype=torch.float32)
                .to("cuda")
                .unsqueeze(0)
            )
            train_data.set_velocity_per_8_frames(
                torch.tensor(norm_motion["velocity_per_8_frames"], dtype=torch.float32)
                .to("cuda")
                .unsqueeze(0)
            )

            tags_tensor_dict = {
                key: value.clone()
                .detach()
                .unsqueeze(
                    0
                )  # torch.tensor(value, dtype=torch.float32).to('cuda').unsqueeze(0)
                for key, value in norm_motion["tags"].items()
            }
            train_data.set_tags(tags_tensor_dict)
            train_data.set_energy(
                norm_motion["energy_feet"]
                .clone()
                .detach()  # torch.tensor(norm_motion["energy_feet"],dtype=torch.float32).to('cuda').unsqueeze(0)
            )

            if sparse_motions is not None:
                train_data.set_sparse_motion(sparse_motions[index])
            bvh, filename = dataset.get_bvh(index)
            train_data.set_rot_order(bvh.data["rot_order"])
            train_data.set_global_pos(
                torch.tensor(bvh.data["positions"][:, 0]).to("cuda").unsqueeze(0)
            )
            res, disp_8, yaw_diff = generator_model.forward()
            results.append((res, bvh, filename))
            disps.append(disp_8)
            yaw_diffs.append(yaw_diff)

    return results, disps, yaw_diffs


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
                norm_motion["disp_8"].unsqueeze(0),
                norm_motion["tags"]["sin_diff"].unsqueeze(0),
                norm_motion["tags"]["cos_diff"].unsqueeze(0),
                norm_motion["loss_weights"].unsqueeze(0),
            )
            res = ik_model.forward(results_decoder[index][0])
            bvh, filename = dataset.get_bvh(index)
            results.append((res, bvh, filename))
    return results
