import json
from pathlib import Path

import kornia
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from models.model_utilities import *
from utilities.utilities import *

CLASS_LABELS = {0: "No water", 1: "Permanent Waters", 2: "Floods", 3: "Invalid pixels"}


def train_semantic_segmentation(
    model, train_loader, val_loader, test_loader, configs, model_configs
):
    if configs["wandb_activate"]:
        # Store wandb id to continue run
        id = wandb.util.generate_id()
        json.dump({"run_id": id}, open(configs["checkpoint_path"] + "/id.json", "w"))
        wandb.init(
            project=configs["wandb_project"],
            entity=configs["wandb_entity"],
            config=configs,
            id=id,
            resume="allow",
        )
        wandb.watch(model, log_freq=20)

    # Accuracy, loss, optimizer, lr scheduler
    accuracy, fscore, precision, recall, iou = initialize_metrics(configs)

    criterion = create_loss(configs, mode="train")
    optimizer = torch.optim.Adam(model.parameters(), lr=model_configs["learning_rate"])
    lr_scheduler = init_lr_scheduler(
        optimizer, configs, model_configs, steps=len(train_loader)
    )

    model.to(configs["device"])
    best_val = 0.0
    best_stats = {}

    if configs["mixed_precision"]:
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(configs["epochs"]):
        model.train()

        train_loss = 0.0

        for index, batch in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc="Epoch " + str(epoch),
        ):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=configs["mixed_precision"]):
                if configs["scale_input"] is not None:
                    if not configs["dem"]:
                        (
                            image_scale_var_1,
                            image_scale_var_2,
                            image,
                            mask,
                            pre_scale_var_1,
                            pre_scale_var_2,
                            pre_event,
                            pre2_scale_var_1,
                            pre2_scale_var_2,
                            pre_event_2,
                            clz,
                            activation,
                        ) = batch
                    else:
                        (
                            image_scale_var_1,
                            image_scale_var_2,
                            image,
                            mask,
                            pre_scale_var_1,
                            pre_scale_var_2,
                            pre_event,
                            pre2_scale_var_1,
                            pre2_scale_var_2,
                            pre_event_2,
                            dem,
                            clz,
                            activation,
                        ) = batch
                else:
                    if not configs["dem"]:
                        image, mask, pre_event, pre_event_2, clz, activation = batch
                    else:
                        (
                            image,
                            mask,
                            pre_event,
                            pre_event_2,
                            dem,
                            clz,
                            activation,
                        ) = batch

                image = image.to(configs["device"])
                mask = mask.to(configs["device"])
                if configs["dem"]:
                    dem = dem.to(configs["device"])
                    image = torch.cat((image, dem), dim=1)
                if configs["inputs"] == ["post_event"]:
                    output = model(image)
                elif set(configs["inputs"]) == set(["pre_event_1", "post_event"]):
                    pre_event = pre_event.to(configs["device"])
                    image = torch.cat((image, pre_event), dim=1)
                    output = model(image)
                elif set(configs["inputs"]) == set(["pre_event_2", "post_event"]):
                    pre_event_2 = pre_event_2.to(configs["device"])
                    image = torch.cat((image, pre_event_2), dim=1)
                    output = model(image)
                elif (
                    model_configs["architecture"] == "vivit"
                    or model_configs["architecture"] == "convlstm"
                ):
                    pre_event = pre_event.to(configs["device"])
                    pre_event = torch.cat(
                        (
                            pre_event_2.to(configs["device"]).unsqueeze(1),
                            pre_event.unsqueeze(1),
                        ),
                        dim=1,
                    )
                    image = torch.cat(
                        (pre_event, image.unsqueeze(1).to(configs["device"])), dim=1
                    )
                    output = model(image)
                elif set(configs["inputs"]) == set(
                    ["pre_event_1", "pre_event_2", "post_event"]
                ):
                    pre_event = pre_event.to(configs["device"])
                    image = torch.cat((image, pre_event), dim=1)
                    image = torch.cat((image, pre_event_2.to(configs["device"])), dim=1)
                    output = model(image)
                else:
                    print('Invalid configuration for "inputs". Exiting...')
                    exit(1)

                if configs["method"] == "contrastive":
                    out = output.clone()
                    output = out[:, : configs["num_classes"]]

                predictions = output.argmax(1)
                loss = criterion(output, mask)

                train_loss += loss.item() * image.size(0)

            if configs["mixed_precision"]:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            acc = accuracy(predictions, mask)
            score = fscore(predictions, mask)
            prec = precision(predictions, mask)
            rec = recall(predictions, mask)
            ious = iou(predictions, mask)
            mean_iou = (ious[0] + ious[1] + ious[2]) / 3

            if index % configs["print_frequency"] == 0:
                if configs["on_screen_prints"]:
                    print(f"Epoch: {epoch}")
                    print(f"Iteration: {index}")
                    print(f"Train Loss: {loss.item()}")
                    print(f"Train Accuracy ({CLASS_LABELS[0]}): {100 * acc[0].item()}")
                    print(f"Train Accuracy ({CLASS_LABELS[1]}): {100 * acc[1].item()}")
                    print(f"Train Accuracy ({CLASS_LABELS[2]}): {100 * acc[2].item()}")
                    print(f"Train F-Score ({CLASS_LABELS[0]}): {100 * score[0].item()}")
                    print(f"Train F-Score ({CLASS_LABELS[1]}): {100 * score[1].item()}")
                    print(f"Train F-Score ({CLASS_LABELS[2]}): {100 * score[2].item()}")
                    print(
                        f"Train Precision ({CLASS_LABELS[0]}): {100 * prec[0].item()}"
                    )
                    print(
                        f"Train Precision ({CLASS_LABELS[1]}): {100 * prec[1].item()}"
                    )
                    print(
                        f"Train Precision ({CLASS_LABELS[2]}): {100 * prec[2].item()}"
                    )
                    print(f"Train Recall ({CLASS_LABELS[0]}): {100 * rec[0].item()}")
                    print(f"Train Recall ({CLASS_LABELS[1]}): {100 * rec[1].item()}")
                    print(f"Train Recall ({CLASS_LABELS[2]}): {100 * rec[2].item()}")
                    print(f"Train IoU ({CLASS_LABELS[0]}): {100 * ious[0].item()}")
                    print(f"Train IoU ({CLASS_LABELS[1]}): {100 * ious[1].item()}")
                    print(f"Train IoU ({CLASS_LABELS[2]}): {100 * ious[2].item()}")
                    print(f"Train MeanIoU: {mean_iou * 100}")
                    print(f"lr: {lr_scheduler.get_last_lr()[0]}")
                elif configs["wandb_activate"]:
                    wandb.log(
                        {
                            "Epoch": epoch,
                            "Iteration": index,
                            "Train Loss": loss.item(),
                            f"Train Accuracy ({CLASS_LABELS[0]})": 100 * acc[0].item(),
                            f"Train Accuracy ({CLASS_LABELS[1]})": 100 * acc[1].item(),
                            f"Train Accuracy ({CLASS_LABELS[2]})": 100 * acc[2].item(),
                            f"Train F-Score ({CLASS_LABELS[0]})": 100 * score[0].item(),
                            f"Train F-Score ({CLASS_LABELS[1]})": 100 * score[1].item(),
                            f"Train F-Score ({CLASS_LABELS[2]})": 100 * score[2].item(),
                            f"Train Precision ({CLASS_LABELS[0]})": 100
                            * prec[0].item(),
                            f"Train Precision ({CLASS_LABELS[1]})": 100
                            * prec[1].item(),
                            f"Train Precision ({CLASS_LABELS[2]})": 100
                            * prec[2].item(),
                            f"Train Recall ({CLASS_LABELS[0]})": 100 * rec[0].item(),
                            f"Train Recall ({CLASS_LABELS[1]})": 100 * rec[1].item(),
                            f"Train Recall ({CLASS_LABELS[2]})": 100 * rec[2].item(),
                            f"Train IoU ({CLASS_LABELS[0]})": 100 * ious[0].item(),
                            f"Train IoU ({CLASS_LABELS[1]})": 100 * ious[1].item(),
                            f"Train IoU ({CLASS_LABELS[2]})": 100 * ious[2].item(),
                            "Train MeanIoU": mean_iou * 100,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }
                    )

        # Update LR scheduler
        lr_scheduler.step()

        # Evaluate on validation set
        model.eval()
        val_acc, val_score, miou = eval_semantic_segmentation(
            model,
            val_loader,
            settype="Val",
            configs=configs,
            model_configs=model_configs,
        )

        if miou > best_val:
            print("Epoch: ", epoch)
            print("New best validation mIOU: ", miou)
            if configs["wandb_activate"]:
                wandb.log({"Best Validation mIOU": miou})
            print(
                "Saving model to: ",
                configs["checkpoint_path"] + "/" + "best_segmentation.pt",
            )
            best_val = miou
            best_stats["miou"] = best_val
            best_stats["epoch"] = epoch
            torch.save(model, configs["checkpoint_path"] + "/" + "best_segmentation.pt")


def eval_semantic_segmentation(
    model, loader, configs=None, settype="Test", model_configs=None
):
    accuracy, fscore, precision, recall, iou = initialize_metrics(configs, mode="val")
    if configs["evaluate_water"]:
        water_fscore = F1Score(
            task="multiclass",
            num_classes=2,
            average="none",
            multidim_average="global",
            ignore_index=3,
        ).to(configs["device"])
    if configs["log_zone_metrics"]:
        (
            accuracy_clzone1,
            fscore_clzone1,
            precision_clzone1,
            recall_clzone1,
            iou_clzone1,
        ) = initialize_metrics(configs, mode="val")
        (
            accuracy_clzone2,
            fscore_clzone2,
            precision_clzone2,
            recall_clzone2,
            iou_clzone2,
        ) = initialize_metrics(configs, mode="val")
        (
            accuracy_clzone3,
            fscore_clzone3,
            precision_clzone3,
            recall_clzone3,
            iou_clzone3,
        ) = initialize_metrics(configs, mode="val")

    if configs["log_AOI_metrics"]:
        activ_metrics = {
            activ: initialize_metrics(configs, mode="val")
            for activ in loader.dataset.activations
        }
        if configs["evaluate_water"]:
            water_only_metrics = {
                activ: F1Score(
                    task="multiclass",
                    num_classes=2,
                    average="none",
                    multidim_average="global",
                    ignore_index=3,
                ).to(configs["device"])
                for activ in loader.dataset.activations
            }

    model.to(configs["device"])
    criterion = create_loss(configs, mode="val")

    first_image = []
    first_mask = []
    first_prediction = []
    total_loss = 0.0

    samples_per_clzone = {1: 0, 2: 0, 3: 0}
    random_index = 0
    for index, batch in tqdm(enumerate(loader), total=len(loader)):
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                if configs["scale_input"] is not None:
                    if not configs["dem"]:
                        (
                            image_scale_var_1,
                            image_scale_var_2,
                            image,
                            mask,
                            pre_scale_var_1,
                            pre_scale_var_2,
                            pre_event,
                            pre2_scale_var_1,
                            pre2_scale_var_2,
                            pre_event_2,
                            clz,
                            activ,
                        ) = batch
                    else:
                        (
                            image_scale_var_1,
                            image_scale_var_2,
                            image,
                            mask,
                            pre_scale_var_1,
                            pre_scale_var_2,
                            pre_event,
                            pre2_scale_var_1,
                            pre2_scale_var_2,
                            pre_event_2,
                            dem,
                            clz,
                            activ,
                        ) = batch
                else:
                    if not configs["dem"]:
                        image, mask, pre_event, pre_event_2, clz, activ = batch
                    else:
                        image, mask, pre_event, pre_event_2, dem, clz, activ = batch

                image = image.to(configs["device"])
                mask = mask.to(configs["device"])
                if configs["dem"]:
                    dem = dem.to(configs["device"])
                    image = torch.cat((image, dem), dim=1)
                elif configs["inputs"] == ["post_event"]:
                    output = model(image)
                elif set(configs["inputs"]) == set(["pre_event_1", "post_event"]):
                    pre_event = pre_event.to(configs["device"])
                    image = torch.cat((image, pre_event), dim=1)
                    output = model(image)
                elif set(configs["inputs"]) == set(["pre_event_2", "post_event"]):
                    pre_event_2 = pre_event_2.to(configs["device"])
                    image = torch.cat((image, pre_event_2), dim=1)
                    output = model(image)
                elif (
                    model_configs["architecture"] == "vivit"
                    or model_configs["architecture"] == "convlstm"
                ):
                    pre_event = pre_event.to(configs["device"])
                    pre_event = torch.cat(
                        (
                            pre_event_2.to(configs["device"]).unsqueeze(1),
                            pre_event.unsqueeze(1),
                        ),
                        dim=1,
                    )
                    image = torch.cat(
                        (pre_event, image.unsqueeze(1).to(configs["device"])), dim=1
                    )
                    output = model(image)
                elif set(configs["inputs"]) == set(
                    ["pre_event_1", "pre_event_2", "post_event"]
                ):
                    pre_event = pre_event.to(configs["device"])
                    image = torch.cat((image, pre_event), dim=1)
                    image = torch.cat((image, pre_event_2.to(configs["device"])), dim=1)
                    output = model(image)
                else:
                    print('Invalid configuration for "inputs". Exiting...')
                    exit(1)

                loss = criterion(output, mask)
                total_loss += loss.item() * image.size(0)
                predictions = output.argmax(1)

                water_only_predictions = predictions.clone()
                water_only_predictions[water_only_predictions == 2] = 1
                water_only_labels = mask.clone()
                water_only_labels[water_only_labels == 2] = 1
                water_fscore(water_only_predictions, water_only_labels)

                if index == random_index:
                    first_image = image.detach().cpu()[0]
                    pre_event_wand = pre_event.detach().cpu()[0]
                    pre_event_2_wand = pre_event_2.detach().cpu()[0]

                    first_mask = mask.detach().cpu()[0]
                    first_prediction = predictions.detach().cpu()[0]

                    if configs["scale_input"] is not None:
                        post_image_scale_vars = [torch.stack((image_scale_var_1[0][0], image_scale_var_1[1][0])), torch.stack((image_scale_var_2[0][0], image_scale_var_2[1][0]))]
                        pre_scale_vars = [torch.stack((pre_scale_var_1[0][0], pre_scale_var_1[1][0])), torch.stack((pre_scale_var_2[0][0], pre_scale_var_2[1][0]))]
                        pre2_scale_vars = [torch.stack((pre2_scale_var_1[0][0], pre2_scale_var_1[1][0])), torch.stack((pre2_scale_var_2[0][0], pre2_scale_var_2[1][0]))]

                accuracy(predictions, mask)
                fscore(predictions, mask)
                precision(predictions, mask)
                recall(predictions, mask)
                iou(predictions, mask)

                if configs["log_zone_metrics"]:
                    clz_in_batch = torch.unique(clz)

                    if 1 in clz_in_batch:
                        accuracy_clzone1(
                            predictions[clz == 1, :, :], mask[clz == 1, :, :]
                        )
                        fscore_clzone1(
                            predictions[clz == 1, :, :], mask[clz == 1, :, :]
                        )
                        precision_clzone1(
                            predictions[clz == 1, :, :], mask[clz == 1, :, :]
                        )
                        recall_clzone1(
                            predictions[clz == 1, :, :], mask[clz == 1, :, :]
                        )
                        iou_clzone1(predictions[clz == 1, :, :], mask[clz == 1, :, :])
                        samples_per_clzone[1] += predictions[clz == 1, :, :].shape[0]

                    if 2 in clz_in_batch:
                        accuracy_clzone2(
                            predictions[clz == 2, :, :], mask[clz == 2, :, :]
                        )
                        fscore_clzone2(
                            predictions[clz == 2, :, :], mask[clz == 2, :, :]
                        )
                        precision_clzone2(
                            predictions[clz == 2, :, :], mask[clz == 2, :, :]
                        )
                        recall_clzone2(
                            predictions[clz == 2, :, :], mask[clz == 2, :, :]
                        )
                        iou_clzone2(predictions[clz == 2, :, :], mask[clz == 2, :, :])
                        samples_per_clzone[2] += predictions[clz == 2, :, :].shape[0]

                    if 3 in clz_in_batch:
                        accuracy_clzone3(
                            predictions[clz == 3, :, :], mask[clz == 3, :, :]
                        )
                        fscore_clzone3(
                            predictions[clz == 3, :, :], mask[clz == 3, :, :]
                        )
                        precision_clzone3(
                            predictions[clz == 3, :, :], mask[clz == 3, :, :]
                        )
                        recall_clzone3(
                            predictions[clz == 3, :, :], mask[clz == 3, :, :]
                        )
                        iou_clzone3(predictions[clz == 3, :, :], mask[clz == 3, :, :])
                        samples_per_clzone[3] += predictions[clz == 3, :, :].shape[0]

                if configs["log_AOI_metrics"]:
                    activs_in_batch = torch.unique(activ)

                    for activ_i in [i.item() for i in activs_in_batch]:
                        activ_metrics[activ_i][0](
                            predictions[activ == activ_i, :, :],
                            mask[activ == activ_i, :, :],
                        )  # accuracy
                        activ_metrics[activ_i][1](
                            predictions[activ == activ_i, :, :],
                            mask[activ == activ_i, :, :],
                        )  # fscore
                        activ_metrics[activ_i][2](
                            predictions[activ == activ_i, :, :],
                            mask[activ == activ_i, :, :],
                        )  # precision
                        activ_metrics[activ_i][3](
                            predictions[activ == activ_i, :, :],
                            mask[activ == activ_i, :, :],
                        )  # recall
                        activ_metrics[activ_i][4](
                            predictions[activ == activ_i, :, :],
                            mask[activ == activ_i, :, :],
                        )  # iou

                        if configs["evaluate_water"]:
                            water_only_metrics[activ_i](
                                water_only_predictions[activ == activ_i, :, :],
                                water_only_labels[activ == activ_i, :, :],
                            )

    # Calculate average loss over an epoch
    val_loss = total_loss / len(loader)
    if configs["wandb_activate"]:
        mask_example = first_mask
        prediction_example = first_prediction

        # Reverse image scaling for visualization purposes
        if (
            configs["scale_input"] not in [None, "custom"]
            and configs["reverse_scaling"]
        ):
            pre_event_wand = reverse_scale_img(
                pre_event_wand, pre_scale_vars[0], pre_scale_vars[1], configs
            )
            pre_event_2_wand = reverse_scale_img(
                pre_event_2_wand, pre2_scale_vars[0], pre2_scale_vars[1], configs
            )
            first_image = reverse_scale_img(
                first_image, post_image_scale_vars[0], post_image_scale_vars[1], configs
            )

        first_image = kornia.enhance.adjust_gamma(first_image, gamma=0.3)
        pre_event_wand = kornia.enhance.adjust_gamma(pre_event_wand, gamma=0.3)
        pre_event_2_wand = kornia.enhance.adjust_gamma(pre_event_2_wand, gamma=0.3)

        if (
            model_configs["architecture"] == "vivit"
            or model_configs["architecture"] == "convlstm"
        ):
            first_image = first_image[2]
            pre_event_wand = pre_event_wand[1]

        mask_img = wandb.Image(
            (first_image[0] * 255).int().cpu().detach().numpy(),
            masks={
                "predictions": {
                    "mask_data": prediction_example.float().numpy(),
                    "class_labels": CLASS_LABELS,
                },
                "ground_truth": {
                    "mask_data": mask_example.float().numpy(),
                    "class_labels": CLASS_LABELS,
                },
            },
        )
        mask_img_preevent_1 = wandb.Image(
            (pre_event_wand[0] * 255).int().cpu().detach().numpy(),
            masks={
                "predictions": {
                    "mask_data": prediction_example.float().numpy(),
                    "class_labels": CLASS_LABELS,
                },
                "ground_truth": {
                    "mask_data": mask_example.float().numpy(),
                    "class_labels": CLASS_LABELS,
                },
            },
        )
        mask_img_preevent_2 = wandb.Image(
            (pre_event_2_wand[0] * 255).int().cpu().detach().numpy(),
            masks={
                "predictions": {
                    "mask_data": prediction_example.float().numpy(),
                    "class_labels": CLASS_LABELS,
                },
                "ground_truth": {
                    "mask_data": mask_example.float().numpy(),
                    "class_labels": CLASS_LABELS,
                },
            },
        )
        wandb.log({settype + " Flood Masks ": mask_img})
        wandb.log({settype + " Pre-event_1 Masks ": mask_img_preevent_1})
        wandb.log({settype + " Pre-event_2 Masks ": mask_img_preevent_2})

    acc = accuracy.compute()
    score = fscore.compute()
    prec = precision.compute()
    rec = recall.compute()
    ious = iou.compute()
    mean_iou = ious[:3].mean()
    if configs["evaluate_water"]:
        water_total_fscore = water_fscore.compute()

    if configs["log_zone_metrics"]:
        acc_clz1 = accuracy_clzone1.compute()
        score_clz1 = fscore_clzone1.compute()
        prec_clz1 = precision_clzone1.compute()
        rec_clz1 = recall_clzone1.compute()
        ious_clz1 = iou_clzone1.compute()
        mean_iou_clz1 = ious_clz1[:3].mean()

        acc_clz2 = accuracy_clzone2.compute()
        score_clz2 = fscore_clzone2.compute()
        prec_clz2 = precision_clzone2.compute()
        rec_clz2 = recall_clzone2.compute()
        ious_clz2 = iou_clzone2.compute()
        mean_iou_clz2 = ious_clz2[:3].mean()

        acc_clz3 = accuracy_clzone3.compute()
        score_clz3 = fscore_clzone3.compute()
        prec_clz3 = precision_clzone3.compute()
        rec_clz3 = recall_clzone3.compute()
        ious_clz3 = iou_clzone3.compute()
        mean_iou_clz3 = ious_clz3[:3].mean()

    if configs["log_AOI_metrics"]:
        activ_i_metrics = {}
        for activ_i, activ_i_metrics_f in activ_metrics.items():
            activ_i_metrics[activ_i] = {}
            activ_i_metrics[activ_i]["accuracy"] = activ_i_metrics_f[
                0
            ].compute()  # accuracy
            activ_i_metrics[activ_i]["fscore"] = activ_i_metrics_f[
                1
            ].compute()  # fscore
            activ_i_metrics[activ_i]["precision"] = activ_i_metrics_f[
                2
            ].compute()  # precision
            activ_i_metrics[activ_i]["recall"] = activ_i_metrics_f[
                3
            ].compute()  # recall
            activ_i_metrics[activ_i]["iou"] = activ_i_metrics_f[4].compute()  # iou

        water_act_metrics = {}
        for activ_i in water_only_metrics.keys():
            water_act_metrics[activ_i] = water_only_metrics[activ_i].compute()

    if configs["on_screen_prints"]:
        print(f'\n{"="*20}')

        print(f"{settype} Loss: {val_loss}")
        print(f"{settype} Accuracy ({CLASS_LABELS[0]}): {100 * acc[0].item()}")
        print(f"{settype} Accuracy ({CLASS_LABELS[1]}): {100 * acc[1].item()}")
        print(f"{settype} Accuracy ({CLASS_LABELS[2]}): {100 * acc[2].item()}")
        print(f"{settype} F-Score ({CLASS_LABELS[0]}): {100 * score[0].item()}")
        print(f"{settype} F-Score ({CLASS_LABELS[1]}): {100 * score[1].item()}")
        print(f"{settype} F-Score ({CLASS_LABELS[2]}): {100 * score[2].item()}")
        print(f"{settype} Precision ({CLASS_LABELS[0]}): {100 * prec[0].item()}")
        print(f"{settype} Precision ({CLASS_LABELS[1]}): {100 * prec[1].item()}")
        print(f"{settype} Precision ({CLASS_LABELS[2]}): {100 * prec[2].item()}")
        print(f"{settype} Recall ({CLASS_LABELS[0]}): {100 * rec[0].item()}")
        print(f"{settype} Recall ({CLASS_LABELS[1]}): {100 * rec[1].item()}")
        print(f"{settype} Recall ({CLASS_LABELS[2]}): {100 * rec[2].item()}")
        print(f"{settype} IoU ({CLASS_LABELS[0]}): {100 * ious[0].item()}")
        print(f"{settype} IoU ({CLASS_LABELS[1]}): {100 * ious[1].item()}")
        print(f"{settype} IoU ({CLASS_LABELS[2]}): {100 * ious[2].item()}")
        print(f"{settype} MeanIoU: {mean_iou * 100}")

        print(f'\n{"="*20}')

        if configs["log_zone_metrics"]:
            print(f'\n{"="*20}\n')
            print("Metrics for climatic zone 1")
            print("Number of samples for climatic zone 1 = ", samples_per_clzone[1])
            print(f'\n{"="*20}')
            print(f"{settype} Accuracy ({CLASS_LABELS[0]}): {100 * acc_clz1[0].item()}")
            print(f"{settype} Accuracy ({CLASS_LABELS[1]}): {100 * acc_clz1[1].item()}")
            print(f"{settype} Accuracy ({CLASS_LABELS[2]}): {100 * acc_clz1[2].item()}")
            print(
                f"{settype} F-Score ({CLASS_LABELS[0]}): {100 * score_clz1[0].item()}"
            )
            print(
                f"{settype} F-Score ({CLASS_LABELS[1]}): {100 * score_clz1[1].item()}"
            )
            print(
                f"{settype} F-Score ({CLASS_LABELS[2]}): {100 * score_clz1[2].item()}"
            )
            print(
                f"{settype} Precision ({CLASS_LABELS[0]}): {100 * prec_clz1[0].item()}"
            )
            print(
                f"{settype} Precision ({CLASS_LABELS[1]}): {100 * prec_clz1[1].item()}"
            )
            print(
                f"{settype} Precision ({CLASS_LABELS[2]}): {100 * prec_clz1[2].item()}"
            )
            print(f"{settype} Recall ({CLASS_LABELS[0]}): {100 * rec_clz1[0].item()}")
            print(f"{settype} Recall ({CLASS_LABELS[1]}): {100 * rec_clz1[1].item()}")
            print(f"{settype} Recall ({CLASS_LABELS[2]}): {100 * rec_clz1[2].item()}")
            print(f"{settype} IoU ({CLASS_LABELS[0]}): {100 * ious_clz1[0].item()}")
            print(f"{settype} IoU ({CLASS_LABELS[1]}): {100 * ious_clz1[1].item()}")
            print(f"{settype} IoU ({CLASS_LABELS[2]}): {100 * ious_clz1[2].item()}")
            print(f"{settype} MeanIoU: {mean_iou_clz1 * 100}")

            print(f'\n{"="*20}')

            print(f'\n{"="*20}\n')
            print("Metrics for climatic zone 2")
            print("Number of samples for climatic zone 2 = ", samples_per_clzone[2])

            print(f'\n{"="*20}')
            print(f"{settype} Accuracy ({CLASS_LABELS[0]}): {100 * acc_clz2[0].item()}")
            print(f"{settype} Accuracy ({CLASS_LABELS[1]}): {100 * acc_clz2[1].item()}")
            print(f"{settype} Accuracy ({CLASS_LABELS[2]}): {100 * acc_clz2[2].item()}")
            print(
                f"{settype} F-Score ({CLASS_LABELS[0]}): {100 * score_clz2[0].item()}"
            )
            print(
                f"{settype} F-Score ({CLASS_LABELS[1]}): {100 * score_clz2[1].item()}"
            )
            print(
                f"{settype} F-Score ({CLASS_LABELS[2]}): {100 * score_clz2[2].item()}"
            )
            print(
                f"{settype} Precision ({CLASS_LABELS[0]}): {100 * prec_clz2[0].item()}"
            )
            print(
                f"{settype} Precision ({CLASS_LABELS[1]}): {100 * prec_clz2[1].item()}"
            )
            print(
                f"{settype} Precision ({CLASS_LABELS[2]}): {100 * prec_clz2[2].item()}"
            )
            print(f"{settype} Recall ({CLASS_LABELS[0]}): {100 * rec_clz2[0].item()}")
            print(f"{settype} Recall ({CLASS_LABELS[1]}): {100 * rec_clz2[1].item()}")
            print(f"{settype} Recall ({CLASS_LABELS[2]}): {100 * rec_clz2[2].item()}")
            print(f"{settype} IoU ({CLASS_LABELS[0]}): {100 * ious_clz2[0].item()}")
            print(f"{settype} IoU ({CLASS_LABELS[1]}): {100 * ious_clz2[1].item()}")
            print(f"{settype} IoU ({CLASS_LABELS[2]}): {100 * ious_clz2[2].item()}")
            print(f"{settype} MeanIoU: {mean_iou_clz2 * 100}")

            print(f'\n{"="*20}')

            print(f'\n{"="*20}\n')
            print("Metrics for climatic zone 3")
            print("Number of samples for climatic zone 3 = ", samples_per_clzone[3])

            print(f'\n{"="*20}')
            print(f"{settype} Accuracy ({CLASS_LABELS[0]}): {100 * acc_clz3[0].item()}")
            print(f"{settype} Accuracy ({CLASS_LABELS[1]}): {100 * acc_clz3[1].item()}")
            print(f"{settype} Accuracy ({CLASS_LABELS[2]}): {100 * acc_clz3[2].item()}")
            print(
                f"{settype} F-Score ({CLASS_LABELS[0]}): {100 * score_clz3[0].item()}"
            )
            print(
                f"{settype} F-Score ({CLASS_LABELS[1]}): {100 * score_clz3[1].item()}"
            )
            print(
                f"{settype} F-Score ({CLASS_LABELS[2]}): {100 * score_clz3[2].item()}"
            )
            print(
                f"{settype} Precision ({CLASS_LABELS[0]}): {100 * prec_clz3[0].item()}"
            )
            print(
                f"{settype} Precision ({CLASS_LABELS[1]}): {100 * prec_clz3[1].item()}"
            )
            print(
                f"{settype} Precision ({CLASS_LABELS[2]}): {100 * prec_clz3[2].item()}"
            )
            print(f"{settype} Recall ({CLASS_LABELS[0]}): {100 * rec_clz3[0].item()}")
            print(f"{settype} Recall ({CLASS_LABELS[1]}): {100 * rec_clz3[1].item()}")
            print(f"{settype} Recall ({CLASS_LABELS[2]}): {100 * rec_clz3[2].item()}")
            print(f"{settype} IoU ({CLASS_LABELS[0]}): {100 * ious_clz3[0].item()}")
            print(f"{settype} IoU ({CLASS_LABELS[1]}): {100 * ious_clz3[1].item()}")
            print(f"{settype} IoU ({CLASS_LABELS[2]}): {100 * ious_clz3[2].item()}")
            print(f"{settype} MeanIoU: {mean_iou_clz3 * 100}")

            print(f'\n{"="*20}')

        if configs["log_AOI_metrics"]:
            for activ_i, activ_i_metrics_list in activ_i_metrics.items():
                print(f'\n{"="*20}\n')
                print(f"Metrics for AOI {activ_i}")
                print(f'\n{"="*20}')
                print(
                    f'{settype} AOI {activ_i} Accuracy ({CLASS_LABELS[0]}): {100 * activ_i_metrics_list["accuracy"][0].item()}'
                )
                print(
                    f'{settype} AOI {activ_i} Accuracy ({CLASS_LABELS[1]}): {100 * activ_i_metrics_list["accuracy"][1].item()}'
                )
                print(
                    f'{settype} AOI {activ_i} Accuracy ({CLASS_LABELS[2]}): {100 * activ_i_metrics_list["accuracy"][2].item()}'
                )
                print(
                    f'{settype} AOI {activ_i} F-Score ({CLASS_LABELS[0]}): {100 * activ_i_metrics_list["fscore"][0].item()}'
                )
                print(
                    f'{settype} AOI {activ_i} F-Score ({CLASS_LABELS[1]}): {100 * activ_i_metrics_list["fscore"][1].item()}'
                )
                print(
                    f'{settype} AOI {activ_i} F-Score ({CLASS_LABELS[2]}): {100 * activ_i_metrics_list["fscore"][2].item()}'
                )
                print(
                    f'{settype} AOI {activ_i} Precision ({CLASS_LABELS[0]}): {100 * activ_i_metrics_list["precision"][0].item()}'
                )
                print(
                    f'{settype} AOI {activ_i} Precision ({CLASS_LABELS[1]}): {100 * activ_i_metrics_list["precision"][1].item()}'
                )
                print(
                    f'{settype} AOI {activ_i} Precision ({CLASS_LABELS[2]}): {100 * activ_i_metrics_list["precision"][2].item()}'
                )
                print(
                    f'{settype} AOI {activ_i} Recall ({CLASS_LABELS[0]}): {100 * activ_i_metrics_list["recall"][0].item()}'
                )
                print(
                    f'{settype} AOI {activ_i} Recall ({CLASS_LABELS[1]}): {100 * activ_i_metrics_list["recall"][1].item()}'
                )
                print(
                    f'{settype} AOI {activ_i} Recall ({CLASS_LABELS[2]}): {100 * activ_i_metrics_list["recall"][2].item()}'
                )
                print(
                    f'{settype} AOI {activ_i} IoU ({CLASS_LABELS[0]}): {100 * activ_i_metrics_list["iou"][0].item()}'
                )
                print(
                    f'{settype} AOI {activ_i} IoU ({CLASS_LABELS[1]}): {100 * activ_i_metrics_list["iou"][1].item()}'
                )
                print(
                    f'{settype} AOI {activ_i} IoU ({CLASS_LABELS[2]}): {100 * activ_i_metrics_list["iou"][2].item()}'
                )
                print(
                    f'{settype} AOI {activ_i} MeanIoU: {activ_i_metrics_list["iou"][:3].mean() * 100}'
                )

                print(f'\n{"="*20}')

    elif configs["wandb_activate"]:
        log_dict = {
            f"{settype} Loss": val_loss,
            f"{settype} Accuracy ({CLASS_LABELS[0]})": 100 * acc[0].item(),
            f"{settype} Accuracy ({CLASS_LABELS[1]})": 100 * acc[1].item(),
            f"{settype} Accuracy ({CLASS_LABELS[2]})": 100 * acc[2].item(),
            f"{settype} F-Score ({CLASS_LABELS[0]})": 100 * score[0].item(),
            f"{settype} F-Score ({CLASS_LABELS[1]})": 100 * score[1].item(),
            f"{settype} F-Score ({CLASS_LABELS[2]})": 100 * score[2].item(),
            f"{settype} Precision ({CLASS_LABELS[0]})": 100 * prec[0].item(),
            f"{settype} Precision ({CLASS_LABELS[1]})": 100 * prec[1].item(),
            f"{settype} Precision ({CLASS_LABELS[2]})": 100 * prec[2].item(),
            f"{settype} Recall ({CLASS_LABELS[0]})": 100 * rec[0].item(),
            f"{settype} Recall ({CLASS_LABELS[1]})": 100 * rec[1].item(),
            f"{settype} Recall ({CLASS_LABELS[2]})": 100 * rec[2].item(),
            f"{settype} IoU ({CLASS_LABELS[0]})": 100 * ious[0].item(),
            f"{settype} IoU ({CLASS_LABELS[1]})": 100 * ious[1].item(),
            f"{settype} IoU ({CLASS_LABELS[2]})": 100 * ious[2].item(),
            f"{settype} MeanIoU": mean_iou * 100,
        }
        if configs["evaluate_water"]:
            print("Updating onlywater")

            log_dict.update(
                {
                    f"{settype} OnlyWater-NoWater F-Score ": water_total_fscore[0]
                    * 100,
                    f"{settype} OnlyWater-Water F-Score ": water_total_fscore[1] * 100,
                }
            )

        if configs["log_zone_metrics"]:
            log_dict.update(
                {
                    f"{settype} CLZ 1 Accuracy ({CLASS_LABELS[0]})": 100
                    * acc_clz1[0].item(),
                    f"{settype} CLZ 1 Accuracy ({CLASS_LABELS[1]})": 100
                    * acc_clz1[1].item(),
                    f"{settype} CLZ 1 Accuracy ({CLASS_LABELS[2]})": 100
                    * acc_clz1[2].item(),
                    f"{settype} CLZ 1 F-Score ({CLASS_LABELS[0]})": 100
                    * score_clz1[0].item(),
                    f"{settype} CLZ 1 F-Score ({CLASS_LABELS[1]})": 100
                    * score_clz1[1].item(),
                    f"{settype} CLZ 1 F-Score ({CLASS_LABELS[2]})": 100
                    * score_clz1[2].item(),
                    f"{settype} CLZ 1 Precision ({CLASS_LABELS[0]})": 100
                    * prec_clz1[0].item(),
                    f"{settype} CLZ 1 Precision ({CLASS_LABELS[1]})": 100
                    * prec_clz1[1].item(),
                    f"{settype} CLZ 1 Precision ({CLASS_LABELS[2]})": 100
                    * prec_clz1[2].item(),
                    f"{settype} CLZ 1 Recall ({CLASS_LABELS[0]})": 100
                    * rec_clz1[0].item(),
                    f"{settype} CLZ 1 Recall ({CLASS_LABELS[1]})": 100
                    * rec_clz1[1].item(),
                    f"{settype} CLZ 1 Recall ({CLASS_LABELS[2]})": 100
                    * rec_clz1[2].item(),
                    f"{settype} CLZ 1 IoU ({CLASS_LABELS[0]})": 100
                    * ious_clz1[0].item(),
                    f"{settype} CLZ 1 IoU ({CLASS_LABELS[1]})": 100
                    * ious_clz1[1].item(),
                    f"{settype} CLZ 1 IoU ({CLASS_LABELS[2]})": 100
                    * ious_clz1[2].item(),
                    f"{settype} CLZ 1 MeanIoU": mean_iou_clz1 * 100,
                    f"{settype} CLZ 2 Accuracy ({CLASS_LABELS[0]})": 100
                    * acc_clz2[0].item(),
                    f"{settype} CLZ 2 Accuracy ({CLASS_LABELS[1]})": 100
                    * acc_clz2[1].item(),
                    f"{settype} CLZ 2 Accuracy ({CLASS_LABELS[2]})": 100
                    * acc_clz2[2].item(),
                    f"{settype} CLZ 2 F-Score ({CLASS_LABELS[0]})": 100
                    * score_clz2[0].item(),
                    f"{settype} CLZ 2 F-Score ({CLASS_LABELS[1]})": 100
                    * score_clz2[1].item(),
                    f"{settype} CLZ 2 F-Score ({CLASS_LABELS[2]})": 100
                    * score_clz2[2].item(),
                    f"{settype} CLZ 2 Precision ({CLASS_LABELS[0]})": 100
                    * prec_clz2[0].item(),
                    f"{settype} CLZ 2 Precision ({CLASS_LABELS[1]})": 100
                    * prec_clz2[1].item(),
                    f"{settype} CLZ 2 Precision ({CLASS_LABELS[2]})": 100
                    * prec_clz2[2].item(),
                    f"{settype} CLZ 2 Recall ({CLASS_LABELS[0]})": 100
                    * rec_clz2[0].item(),
                    f"{settype} CLZ 2 Recall ({CLASS_LABELS[1]})": 100
                    * rec_clz2[1].item(),
                    f"{settype} CLZ 2 Recall ({CLASS_LABELS[2]})": 100
                    * rec_clz2[2].item(),
                    f"{settype} CLZ 2 IoU ({CLASS_LABELS[0]})": 100
                    * ious_clz2[0].item(),
                    f"{settype} CLZ 2 IoU ({CLASS_LABELS[1]})": 100
                    * ious_clz2[1].item(),
                    f"{settype} CLZ 2 IoU ({CLASS_LABELS[2]})": 100
                    * ious_clz2[2].item(),
                    f"{settype} CLZ 2 MeanIoU": mean_iou_clz2 * 100,
                    f"{settype} CLZ 3 Accuracy ({CLASS_LABELS[0]})": 100
                    * acc_clz3[0].item(),
                    f"{settype} CLZ 3 Accuracy ({CLASS_LABELS[1]})": 100
                    * acc_clz3[1].item(),
                    f"{settype} CLZ 3 Accuracy ({CLASS_LABELS[2]})": 100
                    * acc_clz3[2].item(),
                    f"{settype} CLZ 3 F-Score ({CLASS_LABELS[0]})": 100
                    * score_clz3[0].item(),
                    f"{settype} CLZ 3 F-Score ({CLASS_LABELS[1]})": 100
                    * score_clz3[1].item(),
                    f"{settype} CLZ 3 F-Score ({CLASS_LABELS[2]})": 100
                    * score_clz3[2].item(),
                    f"{settype} CLZ 3 Precision ({CLASS_LABELS[0]})": 100
                    * prec_clz3[0].item(),
                    f"{settype} CLZ 3 Precision ({CLASS_LABELS[1]})": 100
                    * prec_clz3[1].item(),
                    f"{settype} CLZ 3 Precision ({CLASS_LABELS[2]})": 100
                    * prec_clz3[2].item(),
                    f"{settype} CLZ 3 Recall ({CLASS_LABELS[0]})": 100
                    * rec_clz3[0].item(),
                    f"{settype} CLZ 3 Recall ({CLASS_LABELS[1]})": 100
                    * rec_clz3[1].item(),
                    f"{settype} CLZ 3 Recall ({CLASS_LABELS[2]})": 100
                    * rec_clz3[2].item(),
                    f"{settype} CLZ 3 IoU ({CLASS_LABELS[0]})": 100
                    * ious_clz3[0].item(),
                    f"{settype} CLZ 3 IoU ({CLASS_LABELS[1]})": 100
                    * ious_clz3[1].item(),
                    f"{settype} CLZ 3 IoU ({CLASS_LABELS[2]})": 100
                    * ious_clz3[2].item(),
                    f"{settype} CLZ 3 MeanIoU": mean_iou_clz3 * 100,
                }
            )

        water_log = {}
        if configs["log_AOI_metrics"]:
            for activ_i, activ_i_metrics_list in activ_i_metrics.items():
                log_dict.update(
                    {
                        f"{settype} AOI {activ_i} Accuracy ({CLASS_LABELS[0]})": 100
                        * activ_i_metrics_list["accuracy"][0].item(),
                        f"{settype} AOI {activ_i} Accuracy ({CLASS_LABELS[1]})": 100
                        * activ_i_metrics_list["accuracy"][1].item(),
                        f"{settype} AOI {activ_i} Accuracy ({CLASS_LABELS[2]})": 100
                        * activ_i_metrics_list["accuracy"][2].item(),
                        f"{settype} AOI {activ_i} F-Score ({CLASS_LABELS[0]})": 100
                        * activ_i_metrics_list["fscore"][0].item(),
                        f"{settype} AOI {activ_i} F-Score ({CLASS_LABELS[1]})": 100
                        * activ_i_metrics_list["fscore"][1].item(),
                        f"{settype} AOI {activ_i} F-Score ({CLASS_LABELS[2]})": 100
                        * activ_i_metrics_list["fscore"][2].item(),
                        f"{settype} AOI {activ_i} Precision ({CLASS_LABELS[0]})": 100
                        * activ_i_metrics_list["precision"][0].item(),
                        f"{settype} AOI {activ_i} Precision ({CLASS_LABELS[1]})": 100
                        * activ_i_metrics_list["precision"][1].item(),
                        f"{settype} AOI {activ_i} Precision ({CLASS_LABELS[2]})": 100
                        * activ_i_metrics_list["precision"][2].item(),
                        f"{settype} AOI {activ_i} Recall ({CLASS_LABELS[0]})": 100
                        * activ_i_metrics_list["recall"][0].item(),
                        f"{settype} AOI {activ_i} Recall ({CLASS_LABELS[1]})": 100
                        * activ_i_metrics_list["recall"][1].item(),
                        f"{settype} AOI {activ_i} Recall ({CLASS_LABELS[2]})": 100
                        * activ_i_metrics_list["recall"][2].item(),
                        f"{settype} AOI {activ_i} IoU ({CLASS_LABELS[0]})": 100
                        * activ_i_metrics_list["iou"][0].item(),
                        f"{settype} AOI {activ_i} IoU ({CLASS_LABELS[1]})": 100
                        * activ_i_metrics_list["iou"][1].item(),
                        f"{settype} AOI {activ_i} IoU ({CLASS_LABELS[2]})": 100
                        * activ_i_metrics_list["iou"][2].item(),
                        f"{settype} AOI {activ_i} MeanIoU": activ_i_metrics_list["iou"][
                            :3
                        ].mean()
                        * 100,
                    }
                )

                if configs["evaluate_water"]:
                    water_log.update(
                        {
                            f"{settype} AOI {activ_i} F-Score Only Water": 100
                            * water_act_metrics[activ_i][1].item()
                        }
                    )
                    log_dict.update(water_log)
        wandb.log(log_dict)

    return 100 * acc, 100 * score[:3].mean(), 100 * mean_iou
