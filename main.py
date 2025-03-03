import argparse
import pyjson5 as json
import pprint
import random
from pathlib import Path

import numpy as np
import torch
import wandb
from models.model_utilities import *
from torchmetrics import Accuracy, F1Score, Precision, Recall

import training.train_mae
from training.change_detection_trainer import (
    eval_change_detection,
    train_change_detection,
)
from training.segmentation_trainer import (
    eval_semantic_segmentation,
    train_semantic_segmentation,
)
from training.recurrent_trainer import (
    eval_recurrent_segmentation,
    train_recurrent_segmentation
)
from utilities.utilities import *


parser = argparse.ArgumentParser()
parser.add_argument("--method", default=None)
parser.add_argument("--backbone", default=None)
parser.add_argument("--dem", action='store_true', default=False)
parser.add_argument("--slope", action='store_true', default=False)
parser.add_argument("--batch_size", default=None)
parser.add_argument("--inputs", nargs="+", default=None)
parser.add_argument("--seed", type=int, default=999)


args = parser.parse_args()


# Seed stuff
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

if __name__ == "__main__":
    configs = json.load(open("configs/config.json", "r"))
    if args.method is not None:
        configs["method"] = args.method
    if configs["method"] == "convlstm":
        model_configs = json.load(open("configs/method/temporal/convlstm.json", "r"))
    elif configs["method"] == "vivit":
        model_configs = json.load(open("configs/method/temporal/vivit.json", "r"))
    else:
        model_configs = json.load(
            open(
                f'configs/method/{configs["method"].lower()}/{configs["method"].lower().replace("-", "_")}.json'
            )
        )
        if args.backbone is not None:
            model_configs["backbone"] = args.backbone

    configs.update(model_configs)

    if args.inputs is None and args.dem:
        configs = update_config(configs, None)
    else:
        configs = update_config(configs, args)

    checkpoint_path = create_checkpoint_directory(configs, model_configs)

    if args.batch_size is not None:
        configs["batch_size"] = int(args.batch_size)

    configs["checkpoint_path"] = checkpoint_path
    pprint.pprint(configs)

    # Create Loaders
    train_loader, val_loader, test_loader = prepare_loaders(configs)

    # Begin Training
    if configs["task"] == "segmentation":
        if configs['method'] == 'convlstm':
            if not configs['test']:
                model = initialize_recurrent_model(configs, model_configs)

                train_recurrent_segmentation(
                    model,
                    train_loader,
                    val_loader,
                    test_loader,
                    configs=configs,
                    model_configs=model_configs,
                )

            # Evaluate on Test Set
            model = initialize_recurrent_model(configs, model_configs)

            ckpt_path = Path(configs["checkpoint_path"]) / "best_segmentation.pt"

            print(f"Loading model from: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=configs['device'])
            model.load_state_dict(checkpoint["model_state_dict"])

            test_acc, test_score, miou = eval_recurrent_segmentation(
                model,
                test_loader,
                ckpt_path.parent,
                settype="Test",
                configs=configs,
                model_configs=model_configs,
            )

            # Print final results
            print("Test Mean IOU: ", miou)
        else:
            # Create model
            model = initialize_segmentation_model(configs, model_configs)
            if not configs["test"]:
                train_semantic_segmentation(
                    model,
                    train_loader,
                    val_loader,
                    test_loader,
                    configs=configs,
                    model_configs=model_configs,
                )
            else:
                if configs["wandb_activate"]:
                    # Store wandb id to continue run

                    id = wandb.util.generate_id()
                    json.dump(
                        {"run_id": id}, open(configs["checkpoint_path"] + "/id.json", "w")
                    )
                    wandb.init(
                        project=configs["wandb_project"],
                        entity=configs["wandb_entity"],
                        config=configs,
                        id=id,
                        resume="allow",
                    )
                    wandb.watch(model, log_freq=20)

            # Evaluate on Test Set
            print(
                "Loading model from: ",
                configs["checkpoint_path"] + "/" + "best_segmentation.pt",
            )
            model = torch.load(configs["checkpoint_path"] + "/" + "best_segmentation.pt")
            test_acc, test_score, miou = eval_semantic_segmentation(
                model,
                test_loader,
                settype="Test",
                configs=configs,
                model_configs=model_configs,
            )
            print("Test Mean IOU: ", miou)
    elif configs["task"] == "mae":
        print("Initializing Self-Supervised learning training with configs:")
        pprint.pprint(configs)
        training.train_mae.train(configs)
    elif configs["task"] == "cd":
        model = initialize_cd_model(configs, model_configs, "train")

        train_change_detection(
            model,
            train_loader,
            val_loader,
            test_loader,
            configs=configs,
            model_configs=model_configs,
        )

        # Evaluate on Test Set
        print(
            "Loading model from: ",
            configs["checkpoint_path"] + "/" + "best_segmentation.pt",
        )

        checkpoint = torch.load(
            configs["checkpoint_path"] + "/" + "best_segmentation.pt", map_location=configs['device']
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        test_acc, test_score, miou = eval_change_detection(
            model,
            test_loader,
            settype="Test",
            configs=configs,
            model_configs=model_configs,
        )

        print("Test Mean IOU: ", miou.item())
