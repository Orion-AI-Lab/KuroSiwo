import torch
from tqdm import tqdm
from pathlib import Path
import json
import pyjson5
import wandb
import kornia

import torch.nn.functional as F

from utilities.utilities import *
from models.model_utilities import *


CLASS_LABELS = {0: 'No water', 1: 'Permanent Waters', 2: 'Floods', 3: 'Invalid pixels'}


def train_change_detection(model, train_loader, val_loader, test_loader, configs, model_configs):
    assert len(configs['inputs']) == 2, \
        print(f'Model {model_configs["method"]} requires exactly 2 input images.')

    if configs['wandb_activate']:
        if configs['resume_wandb']:
            wid = json.load(open(f'{configs["checkpoint_path"]}/id.json', 'r'))['run_id']
        else:
            # Store wandb id to continue run
            wid = wandb.util.generate_id()
            pyjson5.dump({'run_id': str(wid)}, open(configs['checkpoint_path'] + '/id.json', 'wb'), quote_keys=True)

        wandb.init(project=configs['wandb_project'], entity=configs['wandb_entity'], config=configs, id=wid, resume="allow")
        wandb.watch(model, log_freq=20)

    # Initialize metrics
    accuracy, fscore, precision, recall, iou = initialize_metrics(configs)

    if configs['log_AOI_metrics']:
        activ_metrics = {activ: initialize_metrics(configs, mode='val') for activ in train_loader.dataset.activations}

    model.to(configs['device'])

    # Initialize loss
    criterion = create_loss(configs, mode='train')

    # Initialize optimizer
    if configs['method'] in ['bit-cd', 'hfa-net']:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=model_configs['learning_rate'],
            momentum=model_configs['momentum'],
            weight_decay=model_configs['weight_decay'])
    else:
        if model_configs['optimizer'] == 'adam':
            # Initialize optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=model_configs['learning_rate'])
        elif model_configs['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=model_configs['learning_rate'],
                betas=model_configs['betas'],
                weight_decay=model_configs['weight_decay'])
        elif model_configs['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=model_configs['learning_rate'],
                momentum=model_configs['momentum'],
                weight_decay=model_configs['weight_decay'])

    # Initialize lr scheduler
    lr_scheduler = init_lr_scheduler(optimizer, configs, model_configs, steps=len(train_loader))

    start_epoch = 0
    last_epoch = configs['epochs']

    best_val = 0.0
    best_stats = {}

    total_train_accuracy, total_train_fscore, total_train_prec, total_train_rec, total_train_iou = 0, 0, 0, 0, 0

    if configs['mixed_precision']:
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler()

    total_iters = 0

    print(f'===== checkpoint_path: {configs["checkpoint_path"]} ====')

    for epoch in range(start_epoch, last_epoch):
        model.train()

        train_loss = 0.0

        with tqdm(initial=0, total=len(train_loader)) as pbar:
            for index, batch in enumerate(train_loader):

                if configs['scale_input'] is not None:
                    if configs['dem']:
                        post_scale_var_1, post_scale_var_2, post_event, mask, pre1_scale_var_1, \
                        pre1_scale_var_2, pre_event_1, pre2_scale_var_1, pre2_scale_var_2, pre_event_2, dem, clz, activ = batch
                    else:
                        post_scale_var_1, post_scale_var_2, post_event, mask, pre1_scale_var_1, \
                        pre1_scale_var_2, pre_event_1, pre2_scale_var_1, pre2_scale_var_2, pre_event_2, clz, activ = batch
                else:
                    if configs['dem']:
                        post_event, mask, pre_event_1, pre_event_2, dem, clz, activ = batch
                    else:
                        post_event, mask, pre_event_1, pre_event_2, clz, activ = batch

                with torch.cuda.amp.autocast(enabled=configs['mixed_precision']):
                    pre_event_1 = pre_event_1.to(configs['device'])
                    pre_event_2 = pre_event_2.to(configs['device'])
                    post_event = post_event.to(configs['device'])
                    mask = mask.to(configs['device'])

                    if configs['dem']:
                        dem = dem.to(configs['device'])

                    inputs = []
                    for inp in configs['inputs']:
                        if inp == 'pre_event_1':
                            if configs['dem']:
                                inputs.append(torch.cat((pre_event_1, dem), dim=1))
                            else:
                                inputs.append(pre_event_1)
                        elif inp == 'pre_event_2':
                            if configs['dem']:
                                inputs.append(torch.cat((pre_event_2, dem), dim=1))
                            else:
                                inputs.append(pre_event_2)
                        elif inp == 'post_event':
                            if configs['dem']:
                                inputs.append(torch.cat((post_event, dem), dim=1))
                            else:
                                inputs.append(post_event)

                    optimizer.zero_grad()
                    output = model(*inputs)

                    if configs['method'] == 'changeformer':
                        if model_configs['multi_scale_infer']:
                            final_output = torch.zeros(output[-1].size()).to(configs['device'])
                            for pred in output:
                                if pred.size(2) != output[-1].size(2):
                                    final_output = final_output + F.interpolate(pred, size=output[-1].size(2), mode="nearest")
                                else:
                                    final_output = final_output + pred
                            final_output = final_output / len(output)
                        else:
                            final_output = output[-1]

                        predictions = final_output.argmax(1)
                    else:
                        predictions = output.argmax(1)

                    if configs['method'] == 'changeformer':
                        if model_configs['multi_scale_train']:
                            i = 0
                            temp_loss = 0.0
                            for pred in output:
                                if pred.size(2) != mask.size(2):
                                    temp_loss = temp_loss + model_configs['multi_pred_weights'][i] * criterion(pred, F.interpolate(mask, size=pred.size(2), mode="nearest"))
                                else:
                                    temp_loss = temp_loss + model_configs['multi_pred_weights'][i] * criterion(pred, mask)
                                i+=1
                            loss = temp_loss
                        else:
                            loss = criterion(output[-1], mask)
                    else:
                        loss = criterion(output, mask)

                    # Note: loss.item() is averaged across all training examples of the current batch
                    # so we multiply by the batch size to obtain the unaveraged current loss
                    train_loss += (loss.item() * pre_event_1.size(0))

                    if configs['mixed_precision']:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    loss_val = loss.item()

                acc = accuracy(predictions, mask)
                score = fscore(predictions, mask)
                prec = precision(predictions, mask)
                rec = recall(predictions, mask)
                ious = iou(predictions, mask)
                mean_iou = ious[:3].mean()

                if configs['log_AOI_metrics']:
                    activs_in_batch = torch.unique(activ)

                    for activ_i in [i.item() for i in activs_in_batch]:
                        activ_metrics[activ_i][0](predictions[activ==activ_i, :, :], mask[activ==activ_i, :, :]) # accuracy
                        activ_metrics[activ_i][1](predictions[activ==activ_i, :, :], mask[activ==activ_i, :, :]) # fscore
                        activ_metrics[activ_i][2](predictions[activ==activ_i, :, :], mask[activ==activ_i, :, :]) # precision
                        activ_metrics[activ_i][3](predictions[activ==activ_i, :, :], mask[activ==activ_i, :, :]) # recall
                        activ_metrics[activ_i][4](predictions[activ==activ_i, :, :], mask[activ==activ_i, :, :]) # iou

                if index % configs['print_frequency'] == 0:
                    pbar.set_description(f'({epoch}) Train Loss: {train_loss:.4f}')

                pbar.update(1)

        if index % configs['train_save_checkpoint_freq'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': loss_val
                }, Path(configs['checkpoint_path']) / f'checkpoint_epoch={epoch}.pt')

        total_train_accuracy = accuracy.compute()
        total_train_fscore = fscore.compute()
        total_train_prec = precision.compute()
        total_train_rec = recall.compute()
        total_train_iou = iou.compute()

        if configs['on_screen_prints'] and (index % configs['print_frequency'] == 0):
            print(f'Epoch: {epoch}')
            print(f'Iteration: {index}')
            print(f'Train Loss: {loss_val}')
            print(f'Train Accuracy ({CLASS_LABELS[0]}): {100 * total_train_accuracy[0].item()}')
            print(f'Train Accuracy ({CLASS_LABELS[1]}): {100 * total_train_accuracy[1].item()}')
            print(f'Train Accuracy ({CLASS_LABELS[2]}): {100 * total_train_accuracy[2].item()}')
            print(f'Train F-Score ({CLASS_LABELS[0]}): {100 * total_train_fscore[0].item()}')
            print(f'Train F-Score ({CLASS_LABELS[1]}): {100 * total_train_fscore[1].item()}')
            print(f'Train F-Score ({CLASS_LABELS[2]}): {100 * total_train_fscore[2].item()}')
            print(f'Train Precision ({CLASS_LABELS[0]}): {100 * total_train_prec[0].item()}')
            print(f'Train Precision ({CLASS_LABELS[1]}): {100 * total_train_prec[1].item()}')
            print(f'Train Precision ({CLASS_LABELS[2]}): {100 * total_train_prec[2].item()}')
            print(f'Train Recall ({CLASS_LABELS[0]}): {100 * total_train_rec[0].item()}')
            print(f'Train Recall ({CLASS_LABELS[1]}): {100 * total_train_rec[1].item()}')
            print(f'Train Recall ({CLASS_LABELS[2]}): {100 * total_train_rec[2].item()}')
            print(f'Train IoU ({CLASS_LABELS[0]}): {100 * total_train_iou[0].item()}')
            print(f'Train IoU ({CLASS_LABELS[1]}): {100 * total_train_iou[1].item()}')
            print(f'Train IoU ({CLASS_LABELS[2]}): {100 * total_train_iou[2].item()}')
            print(f'Train MeanIoU: {mean_iou * 100}')
            print(f'lr: {lr_scheduler.get_last_lr()[0]}')

        elif configs['wandb_activate']:
            log_dict = {
                'Epoch': epoch,
                'Iteration': index,
                'Train Loss': loss_val,
                f'Train Accuracy ({CLASS_LABELS[0]})': 100 * total_train_accuracy[0].item(),
                f'Train Accuracy ({CLASS_LABELS[1]})': 100 * total_train_accuracy[1].item(),
                f'Train Accuracy ({CLASS_LABELS[2]})': 100 * total_train_accuracy[2].item(),
                f'Train F-Score ({CLASS_LABELS[0]})': 100 * total_train_fscore[0].item(),
                f'Train F-Score ({CLASS_LABELS[1]})': 100 * total_train_fscore[1].item(),
                f'Train F-Score ({CLASS_LABELS[2]})': 100 * total_train_fscore[2].item(),
                f'Train Precision ({CLASS_LABELS[0]})': 100 * total_train_prec[0].item(),
                f'Train Precision ({CLASS_LABELS[1]})': 100 * total_train_prec[1].item(),
                f'Train Precision ({CLASS_LABELS[2]})': 100 * total_train_prec[2].item(),
                f'Train Recall ({CLASS_LABELS[0]})': 100 * total_train_rec[0].item(),
                f'Train Recall ({CLASS_LABELS[1]})': 100 * total_train_rec[1].item(),
                f'Train Recall ({CLASS_LABELS[2]})': 100 * total_train_rec[2].item(),
                f'Train IoU ({CLASS_LABELS[0]})': 100 * total_train_iou[0].item(),
                f'Train IoU ({CLASS_LABELS[1]})': 100 * total_train_iou[1].item(),
                f'Train IoU ({CLASS_LABELS[2]})': 100 * total_train_iou[2].item(),
                'Train MeanIoU': mean_iou * 100,
                'lr': lr_scheduler.get_last_lr()[0]
            }

            if configs['log_AOI_metrics']:
                activ_i_metrics = {}
                for activ_i, activ_i_metrics_f in activ_metrics.items():
                    activ_i_metrics[activ_i] = {}
                    activ_i_metrics[activ_i]['accuracy'] = activ_i_metrics_f[0].compute() # accuracy
                    activ_i_metrics[activ_i]['fscore'] = activ_i_metrics_f[1].compute() # fscore
                    activ_i_metrics[activ_i]['precision'] = activ_i_metrics_f[2].compute() # precision
                    activ_i_metrics[activ_i]['recall'] = activ_i_metrics_f[3].compute() # recall
                    activ_i_metrics[activ_i]['iou'] = activ_i_metrics_f[4].compute() # iou

                for activ_i, activ_i_metrics_list in activ_i_metrics.items():
                    log_dict.update({
                        f'Train AOI {activ_i} Accuracy ({CLASS_LABELS[0]})': 100 * activ_i_metrics_list["accuracy"][0].item(),
                        f'Train AOI {activ_i} Accuracy ({CLASS_LABELS[1]})': 100 * activ_i_metrics_list["accuracy"][1].item(),
                        f'Train AOI {activ_i} Accuracy ({CLASS_LABELS[2]})': 100 * activ_i_metrics_list["accuracy"][2].item(),
                        f'Train AOI {activ_i} F-Score ({CLASS_LABELS[0]})': 100 * activ_i_metrics_list["fscore"][0].item(),
                        f'Train AOI {activ_i} F-Score ({CLASS_LABELS[1]})': 100 * activ_i_metrics_list["fscore"][1].item(),
                        f'Train AOI {activ_i} F-Score ({CLASS_LABELS[2]})': 100 * activ_i_metrics_list["fscore"][2].item(),
                        f'Train AOI {activ_i} Precision ({CLASS_LABELS[0]})': 100 * activ_i_metrics_list["precision"][0].item(),
                        f'Train AOI {activ_i} Precision ({CLASS_LABELS[1]})': 100 * activ_i_metrics_list["precision"][1].item(),
                        f'Train AOI {activ_i} Precision ({CLASS_LABELS[2]})': 100 * activ_i_metrics_list["precision"][2].item(),
                        f'Train AOI {activ_i} Recall ({CLASS_LABELS[0]})': 100 * activ_i_metrics_list["recall"][0].item(),
                        f'Train AOI {activ_i} Recall ({CLASS_LABELS[1]})': 100 * activ_i_metrics_list["recall"][1].item(),
                        f'Train AOI {activ_i} Recall ({CLASS_LABELS[2]})': 100 * activ_i_metrics_list["recall"][2].item(),
                        f'Train AOI {activ_i} IoU ({CLASS_LABELS[0]})': 100 * activ_i_metrics_list["iou"][0].item(),
                        f'Train AOI {activ_i} IoU ({CLASS_LABELS[1]})': 100 * activ_i_metrics_list["iou"][1].item(),
                        f'Train AOI {activ_i} IoU ({CLASS_LABELS[2]})': 100 * activ_i_metrics_list["iou"][2].item(),
                        f'Train AOI {activ_i} MeanIoU': activ_i_metrics_list["iou"][:3].mean() * 100
                    })

            wandb.log(log_dict)

        # Update LR scheduler
        lr_scheduler.step()

        #Evaluate on validation set
        val_acc, val_score, miou = eval_change_detection(model, val_loader, settype='Validation', configs=configs, model_configs=model_configs)

        if miou > best_val:
            print(f'New best validation mIoU: {miou}')
            print(f'Saving model to: {configs["checkpoint_path"]}/best_segmentation.pt')
            best_val = miou
            best_stats['acc'] = best_val
            best_stats['epoch'] = epoch

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': loss_val
                }, Path(configs['checkpoint_path']) / 'best_segmentation.pt')

            with open(Path(configs['checkpoint_path']) / 'best_segmentation.txt', 'w') as f:
                f.write(f'{epoch}\n')
                f.write(f'{miou}')


def eval_change_detection(model, loader, settype, configs=None, model_configs=None):
    accuracy, fscore, precision, recall, iou = initialize_metrics(configs, mode='val')

    if configs['evaluate_water']:
        water_fscore = F1Score(task='multiclass', num_classes=2, average='none', multidim_average='global', ignore_index=3).to(configs['device'])

    if configs['log_zone_metrics']:
        accuracy_clzone1, fscore_clzone1, precision_clzone1, recall_clzone1, iou_clzone1 = initialize_metrics(configs, mode='val')
        accuracy_clzone2, fscore_clzone2, precision_clzone2, recall_clzone2, iou_clzone2 = initialize_metrics(configs, mode='val')
        accuracy_clzone3, fscore_clzone3, precision_clzone3, recall_clzone3, iou_clzone3 = initialize_metrics(configs, mode='val')

    if configs['log_AOI_metrics']:
        activ_metrics = {activ: initialize_metrics(configs, mode='val') for activ in loader.dataset.activations}

    criterion = create_loss(configs, mode='val')

    model.to(configs['device'])

    total_iters = 0
    total_loss = 0.0
    model.eval()

    samples_per_clzone = {1: 0, 2: 0, 3: 0}

    random_index = 0 #random.randint(0,len(loader)-1)
    with tqdm(initial=0, total=len(loader)) as pbar:
        for index, batch in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=False):
                with torch.no_grad():
                    if configs['scale_input'] is not None:
                        if configs['dem']:
                            post_scale_var_1, post_scale_var_2, post_event, mask, pre1_scale_var_1, \
                            pre1_scale_var_2, pre_event_1, pre2_scale_var_1, pre2_scale_var_2, pre_event_2, dem, clz, activ = batch
                        else:
                            post_scale_var_1, post_scale_var_2, post_event, mask, pre1_scale_var_1, \
                            pre1_scale_var_2, pre_event_1, pre2_scale_var_1, pre2_scale_var_2, pre_event_2, clz, activ = batch
                    else:
                        if configs['dem']:
                            post_event, mask, pre_event_1, pre_event_2, dem, clz, activ = batch
                        else:
                            post_event, mask, pre_event_1, pre_event_2, clz, activ = batch

                    pre_event_1 = pre_event_1.to(configs['device'])
                    pre_event_2 = pre_event_2.to(configs['device'])
                    post_event = post_event.to(configs['device'])
                    mask = mask.to(configs['device'])

                    if configs['dem']:
                        dem = dem.to(configs['device'])

                    inputs = []
                    for inp in configs['inputs']:
                        if inp == 'pre_event_1':
                            if configs['dem']:
                                inputs.append(torch.cat((pre_event_1, dem), dim=1))
                            else:
                                inputs.append(pre_event_1)
                        elif inp == 'pre_event_2':
                            if configs['dem']:
                                inputs.append(torch.cat((pre_event_2, dem), dim=1))
                            else:
                                inputs.append(pre_event_2)
                        elif inp == 'post_event':
                            if configs['dem']:
                                inputs.append(torch.cat((post_event, dem), dim=1))
                            else:
                                inputs.append(post_event)

                    output = model(*inputs)

                    if configs['method'] == 'changeformer':
                        output = output[-1]

                    predictions = output.argmax(1)

                    loss = criterion(output, mask)

                    # Note: loss.item() is averaged across all training examples of the current batch
                    # so we multiply by the batch size to obtain the unaveraged current loss
                    total_loss += (loss.item() * pre_event_1.size(0))

                    loss_val = loss.item()

                    accuracy(predictions, mask)
                    fscore(predictions, mask)
                    precision(predictions, mask)
                    recall(predictions, mask)
                    iou(predictions, mask)

                    if configs['evaluate_water']:
                        water_only_predictions = predictions.clone()
                        water_only_predictions[water_only_predictions == 2] = 1
                        water_only_labels = mask.clone()
                        water_only_labels[water_only_labels == 2] = 1
                        water_fscore(water_only_predictions, water_only_labels)

                    if index % configs['print_frequency'] == 0:
                        pbar.set_description(f'{settype} Loss: {total_loss:.4f}')

                    if index == random_index:
                        post_image_wand = post_event[0].detach().cpu()
                        pre_event_1_wand = pre_event_1[0].detach().cpu()
                        pre_event_2_wand = pre_event_2[0].detach().cpu()

                        mask_wand = mask[0].detach().cpu()
                        prediction_wand = predictions[0].detach().cpu()

                        if configs['scale_input'] is not None:
                            post_image_scale_vars = [torch.stack((post_scale_var_1[0][0], post_scale_var_1[1][0])), torch.stack((post_scale_var_2[0][0], post_scale_var_2[1][0]))]
                            pre1_scale_vars = [torch.stack((pre1_scale_var_1[0][0], pre1_scale_var_1[1][0])), torch.stack((pre1_scale_var_2[0][0], pre1_scale_var_2[1][0]))]
                            pre2_scale_vars = [torch.stack((pre2_scale_var_1[0][0], pre2_scale_var_1[1][0])), torch.stack((pre2_scale_var_2[0][0], pre2_scale_var_2[1][0]))]

                    if configs['log_zone_metrics']:
                        clz_in_batch = torch.unique(clz)

                        if 1 in clz_in_batch:
                            accuracy_clzone1(predictions[clz==1, :, :], mask[clz==1, :, :])
                            fscore_clzone1(predictions[clz==1, :, :] ,mask[clz==1, :, :])
                            precision_clzone1(predictions[clz==1, :, :], mask[clz==1, :, :])
                            recall_clzone1(predictions[clz==1, :, :], mask[clz==1, :, :])
                            iou_clzone1(predictions[clz==1, :, :], mask[clz==1, :, :])
                            samples_per_clzone[1] += predictions[clz==1, :, :].shape[0]

                        if 2 in clz_in_batch:
                            accuracy_clzone2(predictions[clz==2, :, :], mask[clz==2, :, :])
                            fscore_clzone2(predictions[clz==2, :, :], mask[clz==2, :, :])
                            precision_clzone2(predictions[clz==2, :, :], mask[clz==2, :, :])
                            recall_clzone2(predictions[clz==2, :, :], mask[clz==2, :, :])
                            iou_clzone2(predictions[clz==2, :, :], mask[clz==2, :, :])
                            samples_per_clzone[2] += predictions[clz==2, :, :].shape[0]

                        if 3 in clz_in_batch:
                            accuracy_clzone3(predictions[clz==3, :, :], mask[clz==3, :, :])
                            fscore_clzone3(predictions[clz==3, :, :], mask[clz==3, :, :])
                            precision_clzone3(predictions[clz==3, :, :], mask[clz==3, :, :])
                            recall_clzone3(predictions[clz==3, :, :], mask[clz==3, :, :])
                            iou_clzone3(predictions[clz==3, :, :], mask[clz==3, :, :])
                            samples_per_clzone[3] += predictions[clz==3, :, :].shape[0]

                    if configs['log_AOI_metrics']:
                        activs_in_batch = torch.unique(activ)

                        for activ_i in [i.item() for i in activs_in_batch]:
                            activ_metrics[activ_i][0](predictions[activ==activ_i, :, :], mask[activ==activ_i, :, :]) # accuracy
                            activ_metrics[activ_i][1](predictions[activ==activ_i, :, :], mask[activ==activ_i, :, :]) # fscore
                            activ_metrics[activ_i][2](predictions[activ==activ_i, :, :], mask[activ==activ_i, :, :]) # precision
                            activ_metrics[activ_i][3](predictions[activ==activ_i, :, :], mask[activ==activ_i, :, :]) # recall
                            activ_metrics[activ_i][4](predictions[activ==activ_i, :, :], mask[activ==activ_i, :, :]) # iou

            pbar.update(1)

    # Calculate average loss over an epoch
    val_loss = total_loss / len(loader)

    if configs['wandb_activate']:
        mask_example = mask_wand
        prediction_example = prediction_wand

        # Reverse image scaling for visualization purposes
        if configs['scale_input'] not in [None, 'custom']:
            pre_event_1_wand = reverse_scale_img(pre_event_1_wand, pre1_scale_vars[0], pre1_scale_vars[1], configs)
            pre_event_2_wand = reverse_scale_img(pre_event_2_wand, pre2_scale_vars[0], pre2_scale_vars[1], configs)
            post_image_wand = reverse_scale_img(post_image_wand, post_image_scale_vars[0], post_image_scale_vars[1], configs)

        post_image_wand = kornia.enhance.adjust_gamma(post_image_wand,gamma=0.3)#kornia.enhance.adjust_brightness(first_image, 0.2)
        pre_event_1_wand = kornia.enhance.adjust_gamma(pre_event_1_wand,gamma=0.3)#kornia.enhance.adjust_brightness(first_image, 0.2)
        pre_event_2_wand = kornia.enhance.adjust_gamma(pre_event_2_wand,gamma=0.3)#kornia.enhance.adjust_brightness(first_image, 0.2)

        mask_img = wandb.Image((post_image_wand[0] * 255).int().cpu().detach().numpy(), masks={
            "predictions": {
                "mask_data": prediction_example.float().numpy(),
                "class_labels": CLASS_LABELS
            },
            "ground_truth": {
                "mask_data": mask_example.float().numpy(),
                "class_labels": CLASS_LABELS
            },
        })
        mask_img_preevent_1 = wandb.Image((pre_event_1_wand[0] * 255).int().cpu().detach().numpy(), masks={
            "predictions": {
                "mask_data": prediction_example.float().numpy(),
                "class_labels": CLASS_LABELS
            },
            "ground_truth": {
                "mask_data": mask_example.float().numpy(),
                "class_labels": CLASS_LABELS
            },
        })
        mask_img_preevent_2 = wandb.Image((pre_event_2_wand[0] * 255).int().cpu().detach().numpy(), masks={
            "predictions": {
                "mask_data": prediction_example.float().numpy(),
                "class_labels": CLASS_LABELS
            },
            "ground_truth": {
                "mask_data": mask_example.float().numpy(),
                "class_labels": CLASS_LABELS
            },
        })
        wandb.log({settype + ' Flood Masks ': mask_img})
        wandb.log({settype + ' Pre-event_1 Masks ': mask_img_preevent_1})
        wandb.log({settype + ' Pre-event_2 Masks ': mask_img_preevent_2})

    acc = accuracy.compute()
    score = fscore.compute()
    prec = precision.compute()
    rec = recall.compute()
    ious = iou.compute()
    mean_iou = ious[:3].mean()

    if configs['evaluate_water']:
        water_total_fscore = water_fscore.compute()

    if configs['log_zone_metrics']:
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

    if configs['log_AOI_metrics']:
        activ_i_metrics = {}
        for activ_i, activ_i_metrics_f in activ_metrics.items():
            activ_i_metrics[activ_i] = {}
            activ_i_metrics[activ_i]['accuracy'] = activ_i_metrics_f[0].compute() # accuracy
            activ_i_metrics[activ_i]['fscore'] = activ_i_metrics_f[1].compute() # fscore
            activ_i_metrics[activ_i]['precision'] = activ_i_metrics_f[2].compute() # precision
            activ_i_metrics[activ_i]['recall'] = activ_i_metrics_f[3].compute() # recall
            activ_i_metrics[activ_i]['iou'] = activ_i_metrics_f[4].compute() # iou

    if configs['on_screen_prints']:
        print(f'\n{"="*20}')

        print(f'{settype} Loss: {val_loss}')
        print(f'{settype} Accuracy ({CLASS_LABELS[0]}): {100 * acc[0].item()}')
        print(f'{settype} Accuracy ({CLASS_LABELS[1]}): {100 * acc[1].item()}')
        print(f'{settype} Accuracy ({CLASS_LABELS[2]}): {100 * acc[2].item()}')
        print(f'{settype} F-Score ({CLASS_LABELS[0]}): {100 * score[0].item()}')
        print(f'{settype} F-Score ({CLASS_LABELS[1]}): {100 * score[1].item()}')
        print(f'{settype} F-Score ({CLASS_LABELS[2]}): {100 * score[2].item()}')
        print(f'{settype} Precision ({CLASS_LABELS[0]}): {100 * prec[0].item()}')
        print(f'{settype} Precision ({CLASS_LABELS[1]}): {100 * prec[1].item()}')
        print(f'{settype} Precision ({CLASS_LABELS[2]}): {100 * prec[2].item()}')
        print(f'{settype} Recall ({CLASS_LABELS[0]}): {100 * rec[0].item()}')
        print(f'{settype} Recall ({CLASS_LABELS[1]}): {100 * rec[1].item()}')
        print(f'{settype} Recall ({CLASS_LABELS[2]}): {100 * rec[2].item()}')
        print(f'{settype} IoU ({CLASS_LABELS[0]}): {100 * ious[0].item()}')
        print(f'{settype} IoU ({CLASS_LABELS[1]}): {100 * ious[1].item()}')
        print(f'{settype} IoU ({CLASS_LABELS[2]}): {100 * ious[2].item()}')
        print(f'{settype} MeanIoU: {mean_iou * 100}')

        if configs['evaluate_water']:
            print(f'{settype} F-Score (Only water): {100 * water_total_fscore[1].item()}')

        print(f'\n{"="*20}')

        if configs['log_zone_metrics']:
            print(f'\n{"="*20}\n')
            print('Metrics for climatic zone 1')
            print('Number of samples for climatic zone 1 = ',samples_per_clzone[1])
            print(f'\n{"="*20}')
            print(f'{settype} Accuracy ({CLASS_LABELS[0]}): {100 * acc_clz1[0].item()}')
            print(f'{settype} Accuracy ({CLASS_LABELS[1]}): {100 * acc_clz1[1].item()}')
            print(f'{settype} Accuracy ({CLASS_LABELS[2]}): {100 * acc_clz1[2].item()}')
            print(f'{settype} F-Score ({CLASS_LABELS[0]}): {100 * score_clz1[0].item()}')
            print(f'{settype} F-Score ({CLASS_LABELS[1]}): {100 * score_clz1[1].item()}')
            print(f'{settype} F-Score ({CLASS_LABELS[2]}): {100 * score_clz1[2].item()}')
            print(f'{settype} Precision ({CLASS_LABELS[0]}): {100 * prec_clz1[0].item()}')
            print(f'{settype} Precision ({CLASS_LABELS[1]}): {100 * prec_clz1[1].item()}')
            print(f'{settype} Precision ({CLASS_LABELS[2]}): {100 * prec_clz1[2].item()}')
            print(f'{settype} Recall ({CLASS_LABELS[0]}): {100 * rec_clz1[0].item()}')
            print(f'{settype} Recall ({CLASS_LABELS[1]}): {100 * rec_clz1[1].item()}')
            print(f'{settype} Recall ({CLASS_LABELS[2]}): {100 * rec_clz1[2].item()}')
            print(f'{settype} IoU ({CLASS_LABELS[0]}): {100 * ious_clz1[0].item()}')
            print(f'{settype} IoU ({CLASS_LABELS[1]}): {100 * ious_clz1[1].item()}')
            print(f'{settype} IoU ({CLASS_LABELS[2]}): {100 * ious_clz1[2].item()}')
            print(f'{settype} MeanIoU: {mean_iou_clz1 * 100}')

            print(f'\n{"="*20}')

            print(f'\n{"="*20}\n')
            print('Metrics for climatic zone 2')
            print('Number of samples for climatic zone 2 = ',samples_per_clzone[2])

            print(f'\n{"="*20}')
            print(f'{settype} Accuracy ({CLASS_LABELS[0]}): {100 * acc_clz2[0].item()}')
            print(f'{settype} Accuracy ({CLASS_LABELS[1]}): {100 * acc_clz2[1].item()}')
            print(f'{settype} Accuracy ({CLASS_LABELS[2]}): {100 * acc_clz2[2].item()}')
            print(f'{settype} F-Score ({CLASS_LABELS[0]}): {100 * score_clz2[0].item()}')
            print(f'{settype} F-Score ({CLASS_LABELS[1]}): {100 * score_clz2[1].item()}')
            print(f'{settype} F-Score ({CLASS_LABELS[2]}): {100 * score_clz2[2].item()}')
            print(f'{settype} Precision ({CLASS_LABELS[0]}): {100 * prec_clz2[0].item()}')
            print(f'{settype} Precision ({CLASS_LABELS[1]}): {100 * prec_clz2[1].item()}')
            print(f'{settype} Precision ({CLASS_LABELS[2]}): {100 * prec_clz2[2].item()}')
            print(f'{settype} Recall ({CLASS_LABELS[0]}): {100 * rec_clz2[0].item()}')
            print(f'{settype} Recall ({CLASS_LABELS[1]}): {100 * rec_clz2[1].item()}')
            print(f'{settype} Recall ({CLASS_LABELS[2]}): {100 * rec_clz2[2].item()}')
            print(f'{settype} IoU ({CLASS_LABELS[0]}): {100 * ious_clz2[0].item()}')
            print(f'{settype} IoU ({CLASS_LABELS[1]}): {100 * ious_clz2[1].item()}')
            print(f'{settype} IoU ({CLASS_LABELS[2]}): {100 * ious_clz2[2].item()}')
            print(f'{settype} MeanIoU: {mean_iou_clz2 * 100}')

            print(f'\n{"="*20}')

            print(f'\n{"="*20}\n')
            print('Metrics for climatic zone 3')
            print('Number of samples for climatic zone 3 = ',samples_per_clzone[3])

            print(f'\n{"="*20}')
            print(f'{settype} Accuracy ({CLASS_LABELS[0]}): {100 * acc_clz3[0].item()}')
            print(f'{settype} Accuracy ({CLASS_LABELS[1]}): {100 * acc_clz3[1].item()}')
            print(f'{settype} Accuracy ({CLASS_LABELS[2]}): {100 * acc_clz3[2].item()}')
            print(f'{settype} F-Score ({CLASS_LABELS[0]}): {100 * score_clz3[0].item()}')
            print(f'{settype} F-Score ({CLASS_LABELS[1]}): {100 * score_clz3[1].item()}')
            print(f'{settype} F-Score ({CLASS_LABELS[2]}): {100 * score_clz3[2].item()}')
            print(f'{settype} Precision ({CLASS_LABELS[0]}): {100 * prec_clz3[0].item()}')
            print(f'{settype} Precision ({CLASS_LABELS[1]}): {100 * prec_clz3[1].item()}')
            print(f'{settype} Precision ({CLASS_LABELS[2]}): {100 * prec_clz3[2].item()}')
            print(f'{settype} Recall ({CLASS_LABELS[0]}): {100 * rec_clz3[0].item()}')
            print(f'{settype} Recall ({CLASS_LABELS[1]}): {100 * rec_clz3[1].item()}')
            print(f'{settype} Recall ({CLASS_LABELS[2]}): {100 * rec_clz3[2].item()}')
            print(f'{settype} IoU ({CLASS_LABELS[0]}): {100 * ious_clz3[0].item()}')
            print(f'{settype} IoU ({CLASS_LABELS[1]}): {100 * ious_clz3[1].item()}')
            print(f'{settype} IoU ({CLASS_LABELS[2]}): {100 * ious_clz3[2].item()}')
            print(f'{settype} MeanIoU: {mean_iou_clz3 * 100}')

            print(f'\n{"="*20}')

        if configs['log_AOI_metrics']:
            for activ_i, activ_i_metrics_list in activ_i_metrics.items():
                print(f'\n{"="*20}\n')
                print(f'Metrics for AOI {activ_i}')
                print(f'\n{"="*20}')
                print(f'{settype} AOI {activ_i} Accuracy ({CLASS_LABELS[0]}): {100 * activ_i_metrics_list["accuracy"][0].item()}')
                print(f'{settype} AOI {activ_i} Accuracy ({CLASS_LABELS[1]}): {100 * activ_i_metrics_list["accuracy"][1].item()}')
                print(f'{settype} AOI {activ_i} Accuracy ({CLASS_LABELS[2]}): {100 * activ_i_metrics_list["accuracy"][2].item()}')
                print(f'{settype} AOI {activ_i} F-Score ({CLASS_LABELS[0]}): {100 * activ_i_metrics_list["fscore"][0].item()}')
                print(f'{settype} AOI {activ_i} F-Score ({CLASS_LABELS[1]}): {100 * activ_i_metrics_list["fscore"][1].item()}')
                print(f'{settype} AOI {activ_i} F-Score ({CLASS_LABELS[2]}): {100 * activ_i_metrics_list["fscore"][2].item()}')
                print(f'{settype} AOI {activ_i} Precision ({CLASS_LABELS[0]}): {100 * activ_i_metrics_list["precision"][0].item()}')
                print(f'{settype} AOI {activ_i} Precision ({CLASS_LABELS[1]}): {100 * activ_i_metrics_list["precision"][1].item()}')
                print(f'{settype} AOI {activ_i} Precision ({CLASS_LABELS[2]}): {100 * activ_i_metrics_list["precision"][2].item()}')
                print(f'{settype} AOI {activ_i} Recall ({CLASS_LABELS[0]}): {100 * activ_i_metrics_list["recall"][0].item()}')
                print(f'{settype} AOI {activ_i} Recall ({CLASS_LABELS[1]}): {100 * activ_i_metrics_list["recall"][1].item()}')
                print(f'{settype} AOI {activ_i} Recall ({CLASS_LABELS[2]}): {100 * activ_i_metrics_list["recall"][2].item()}')
                print(f'{settype} AOI {activ_i} IoU ({CLASS_LABELS[0]}): {100 * activ_i_metrics_list["iou"][0].item()}')
                print(f'{settype} AOI {activ_i} IoU ({CLASS_LABELS[1]}): {100 * activ_i_metrics_list["iou"][1].item()}')
                print(f'{settype} AOI {activ_i} IoU ({CLASS_LABELS[2]}): {100 * activ_i_metrics_list["iou"][2].item()}')
                print(f'{settype} AOI {activ_i} MeanIoU: {activ_i_metrics_list["iou"][:3].mean() * 100}')

                print(f'\n{"="*20}')

    elif configs['wandb_activate']:
        log_dict = {
            f'{settype} Loss': val_loss,
            f'{settype} Accuracy ({CLASS_LABELS[0]})': 100 * acc[0].item(),
            f'{settype} Accuracy ({CLASS_LABELS[1]})': 100 * acc[1].item(),
            f'{settype} Accuracy ({CLASS_LABELS[2]})': 100 * acc[2].item(),
            f'{settype} F-Score ({CLASS_LABELS[0]})': 100 * score[0].item(),
            f'{settype} F-Score ({CLASS_LABELS[1]})': 100 * score[1].item(),
            f'{settype} F-Score ({CLASS_LABELS[2]})': 100 * score[2].item(),
            f'{settype} Precision ({CLASS_LABELS[0]})': 100 * prec[0].item(),
            f'{settype} Precision ({CLASS_LABELS[1]})': 100 * prec[1].item(),
            f'{settype} Precision ({CLASS_LABELS[2]})': 100 * prec[2].item(),
            f'{settype} Recall ({CLASS_LABELS[0]})': 100 * rec[0].item(),
            f'{settype} Recall ({CLASS_LABELS[1]})': 100 * rec[1].item(),
            f'{settype} Recall ({CLASS_LABELS[2]})': 100 * rec[2].item(),
            f'{settype} IoU ({CLASS_LABELS[0]})': 100 * ious[0].item(),
            f'{settype} IoU ({CLASS_LABELS[1]})': 100 * ious[1].item(),
            f'{settype} IoU ({CLASS_LABELS[2]})': 100 * ious[2].item(),
            f'{settype} MeanIoU': mean_iou * 100}

        if configs['evaluate_water']:
            log_dict.update({f'{settype} F-Score (Only water)': 100 * water_total_fscore[1].item()})

        if configs['log_zone_metrics']:
            log_dict.update({
                f'{settype} CLZ 1 Accuracy ({CLASS_LABELS[0]})': 100 * acc_clz1[0].item(),
                f'{settype} CLZ 1 Accuracy ({CLASS_LABELS[1]})': 100 * acc_clz1[1].item(),
                f'{settype} CLZ 1 Accuracy ({CLASS_LABELS[2]})': 100 * acc_clz1[2].item(),
                f'{settype} CLZ 1 F-Score ({CLASS_LABELS[0]})': 100 * score_clz1[0].item(),
                f'{settype} CLZ 1 F-Score ({CLASS_LABELS[1]})': 100 * score_clz1[1].item(),
                f'{settype} CLZ 1 F-Score ({CLASS_LABELS[2]})': 100 * score_clz1[2].item(),
                f'{settype} CLZ 1 Precision ({CLASS_LABELS[0]})': 100 * prec_clz1[0].item(),
                f'{settype} CLZ 1 Precision ({CLASS_LABELS[1]})': 100 * prec_clz1[1].item(),
                f'{settype} CLZ 1 Precision ({CLASS_LABELS[2]})': 100 * prec_clz1[2].item(),
                f'{settype} CLZ 1 Recall ({CLASS_LABELS[0]})': 100 * rec_clz1[0].item(),
                f'{settype} CLZ 1 Recall ({CLASS_LABELS[1]})': 100 * rec_clz1[1].item(),
                f'{settype} CLZ 1 Recall ({CLASS_LABELS[2]})': 100 * rec_clz1[2].item(),
                f'{settype} CLZ 1 IoU ({CLASS_LABELS[0]})': 100 * ious_clz1[0].item(),
                f'{settype} CLZ 1 IoU ({CLASS_LABELS[1]})': 100 * ious_clz1[1].item(),
                f'{settype} CLZ 1 IoU ({CLASS_LABELS[2]})': 100 * ious_clz1[2].item(),
                f'{settype} CLZ 1 MeanIoU': mean_iou_clz1 * 100,

                f'{settype} CLZ 2 Accuracy ({CLASS_LABELS[0]})': 100 * acc_clz2[0].item(),
                f'{settype} CLZ 2 Accuracy ({CLASS_LABELS[1]})': 100 * acc_clz2[1].item(),
                f'{settype} CLZ 2 Accuracy ({CLASS_LABELS[2]})': 100 * acc_clz2[2].item(),
                f'{settype} CLZ 2 F-Score ({CLASS_LABELS[0]})': 100 * score_clz2[0].item(),
                f'{settype} CLZ 2 F-Score ({CLASS_LABELS[1]})': 100 * score_clz2[1].item(),
                f'{settype} CLZ 2 F-Score ({CLASS_LABELS[2]})': 100 * score_clz2[2].item(),
                f'{settype} CLZ 2 Precision ({CLASS_LABELS[0]})': 100 * prec_clz2[0].item(),
                f'{settype} CLZ 2 Precision ({CLASS_LABELS[1]})': 100 * prec_clz2[1].item(),
                f'{settype} CLZ 2 Precision ({CLASS_LABELS[2]})': 100 * prec_clz2[2].item(),
                f'{settype} CLZ 2 Recall ({CLASS_LABELS[0]})': 100 * rec_clz2[0].item(),
                f'{settype} CLZ 2 Recall ({CLASS_LABELS[1]})': 100 * rec_clz2[1].item(),
                f'{settype} CLZ 2 Recall ({CLASS_LABELS[2]})': 100 * rec_clz2[2].item(),
                f'{settype} CLZ 2 IoU ({CLASS_LABELS[0]})': 100 * ious_clz2[0].item(),
                f'{settype} CLZ 2 IoU ({CLASS_LABELS[1]})': 100 * ious_clz2[1].item(),
                f'{settype} CLZ 2 IoU ({CLASS_LABELS[2]})': 100 * ious_clz2[2].item(),
                f'{settype} CLZ 2 MeanIoU': mean_iou_clz2 * 100,

                f'{settype} CLZ 3 Accuracy ({CLASS_LABELS[0]})': 100 * acc_clz3[0].item(),
                f'{settype} CLZ 3 Accuracy ({CLASS_LABELS[1]})': 100 * acc_clz3[1].item(),
                f'{settype} CLZ 3 Accuracy ({CLASS_LABELS[2]})': 100 * acc_clz3[2].item(),
                f'{settype} CLZ 3 F-Score ({CLASS_LABELS[0]})': 100 * score_clz3[0].item(),
                f'{settype} CLZ 3 F-Score ({CLASS_LABELS[1]})': 100 * score_clz3[1].item(),
                f'{settype} CLZ 3 F-Score ({CLASS_LABELS[2]})': 100 * score_clz3[2].item(),
                f'{settype} CLZ 3 Precision ({CLASS_LABELS[0]})': 100 * prec_clz3[0].item(),
                f'{settype} CLZ 3 Precision ({CLASS_LABELS[1]})': 100 * prec_clz3[1].item(),
                f'{settype} CLZ 3 Precision ({CLASS_LABELS[2]})': 100 * prec_clz3[2].item(),
                f'{settype} CLZ 3 Recall ({CLASS_LABELS[0]})': 100 * rec_clz3[0].item(),
                f'{settype} CLZ 3 Recall ({CLASS_LABELS[1]})': 100 * rec_clz3[1].item(),
                f'{settype} CLZ 3 Recall ({CLASS_LABELS[2]})': 100 * rec_clz3[2].item(),
                f'{settype} CLZ 3 IoU ({CLASS_LABELS[0]})': 100 * ious_clz3[0].item(),
                f'{settype} CLZ 3 IoU ({CLASS_LABELS[1]})': 100 * ious_clz3[1].item(),
                f'{settype} CLZ 3 IoU ({CLASS_LABELS[2]})': 100 * ious_clz3[2].item(),
                f'{settype} CLZ 3 MeanIoU': mean_iou_clz3 * 100
            })

        if configs['log_AOI_metrics']:
            for activ_i, activ_i_metrics_list in activ_i_metrics.items():
                log_dict.update({
                    f'{settype} AOI {activ_i} Accuracy ({CLASS_LABELS[0]})': 100 * activ_i_metrics_list["accuracy"][0].item(),
                    f'{settype} AOI {activ_i} Accuracy ({CLASS_LABELS[1]})': 100 * activ_i_metrics_list["accuracy"][1].item(),
                    f'{settype} AOI {activ_i} Accuracy ({CLASS_LABELS[2]})': 100 * activ_i_metrics_list["accuracy"][2].item(),
                    f'{settype} AOI {activ_i} F-Score ({CLASS_LABELS[0]})': 100 * activ_i_metrics_list["fscore"][0].item(),
                    f'{settype} AOI {activ_i} F-Score ({CLASS_LABELS[1]})': 100 * activ_i_metrics_list["fscore"][1].item(),
                    f'{settype} AOI {activ_i} F-Score ({CLASS_LABELS[2]})': 100 * activ_i_metrics_list["fscore"][2].item(),
                    f'{settype} AOI {activ_i} Precision ({CLASS_LABELS[0]})': 100 * activ_i_metrics_list["precision"][0].item(),
                    f'{settype} AOI {activ_i} Precision ({CLASS_LABELS[1]})': 100 * activ_i_metrics_list["precision"][1].item(),
                    f'{settype} AOI {activ_i} Precision ({CLASS_LABELS[2]})': 100 * activ_i_metrics_list["precision"][2].item(),
                    f'{settype} AOI {activ_i} Recall ({CLASS_LABELS[0]})': 100 * activ_i_metrics_list["recall"][0].item(),
                    f'{settype} AOI {activ_i} Recall ({CLASS_LABELS[1]})': 100 * activ_i_metrics_list["recall"][1].item(),
                    f'{settype} AOI {activ_i} Recall ({CLASS_LABELS[2]})': 100 * activ_i_metrics_list["recall"][2].item(),
                    f'{settype} AOI {activ_i} IoU ({CLASS_LABELS[0]})': 100 * activ_i_metrics_list["iou"][0].item(),
                    f'{settype} AOI {activ_i} IoU ({CLASS_LABELS[1]})': 100 * activ_i_metrics_list["iou"][1].item(),
                    f'{settype} AOI {activ_i} IoU ({CLASS_LABELS[2]})': 100 * activ_i_metrics_list["iou"][2].item(),
                    f'{settype} AOI {activ_i} MeanIoU': activ_i_metrics_list["iou"][:3].mean() * 100
                })

        wandb.log(log_dict)

    return 100 * acc, 100 * score[:3].mean(), 100 * mean_iou

