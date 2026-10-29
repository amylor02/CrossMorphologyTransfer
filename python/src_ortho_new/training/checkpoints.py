import os
import torch

PRIOR_PREFIX = "autoencoder.codebook_predictor2."


def load_model(model, model_path, train_data, device, ignore_transform_net=True, reset_prior=False):
    model_name = os.path.basename(model_path)[: -len(".pt")]
    assert model_name == "generator" or model_name == "ik"
    if model_name == "generator":
        data_path = model_path[: -len("generator.pt")] + "data.pt"
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]

        # dont load transform net
        # if ignore_transform_net:
        #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.codebook_predictor')}
        #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.disp')}
        #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.tag')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.codebook_predictor2.yaw')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.root_branch')}
        #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.encoder.ema')}
        #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.encoder.num_embeddings')}
        #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.encoder.logits')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.encoder.num_quantizers')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.input_proj')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.forward_dir_proj')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.root_branch')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.root_upsample')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.ctrl_and_input_proj')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.final_cnn_layer')}
        # state_dict = {k: v for k, v in state_dict.items() if (not 'timing_predictor' in k)}
        # state_dict = {k: v for k, v in state_dict.items() if (not 'timing_decoder' in k)}
        if reset_prior:
            # keep the freshly initialised prior: every other tensor comes from the checkpoint. The complete state dict
            # is built first, so the strict load still checks every key of the model.
            complete = model.state_dict()
            for key in complete:
                if key in state_dict and not key.startswith(PRIOR_PREFIX):
                    complete[key] = state_dict[key]
            model.load_state_dict(complete, strict=True)
        # foot-contact priors have a different input width. only an exact key and shape match is accepted.
        elif getattr(getattr(model, "autoencoder", None), "foot_contact", False):
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as exc:
                raise RuntimeError("foot-contact checkpoint does not match this model: " + str(exc)) from exc
        else:
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
    train_data.set_root_means_stds(means["rots"], stds["rots"])
    train_data.set_sin_cos_means_stds(
        means["yaw_sin"], means["yaw_cos"], stds["yaw_sin"], stds["yaw_cos"]
    )
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


def get_periodic_model_paths(name, train_eval_dir):
    # Paths for the periodic snapshots written every 10 epochs. They go in best_root/, which therefore
    # holds periodic snapshots, not necessarily the best model (save_model writes the best one to the model root).
    model_name = (
        "model_" + name + "_" + os.path.basename(os.path.normpath(train_eval_dir))
    )
    model_dir = os.path.join("models", model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    best_root_dir = os.path.join(model_dir, "best_root")
    if not os.path.exists(best_root_dir):
        os.makedirs(best_root_dir)

    data_path = os.path.join(best_root_dir, "data.pt")
    generator_path = os.path.join(best_root_dir, "generator.pt")
    ik_path = os.path.join(model_dir, "ik.pt")
    return data_path, generator_path, ik_path


# backward compatibility
get_model_paths_shared = get_periodic_model_paths


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


def save_periodic_model(
    generator_model,
    ik_model,
    train_dataset,
    name,
    train_eval_dir,
):
    data_path, generator_path, ik_path = get_periodic_model_paths(name, train_eval_dir)

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


# backward compatibility
save_model_shared = save_periodic_model
