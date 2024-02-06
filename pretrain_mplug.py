'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import os
try:
    import ruamel.yaml as yaml
except:
    import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_pretrain_mplug import MPLUG
from models.vit import interpolate_pos_embed, resize_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer, create_two_optimizer
import random
import math
from TPMix import TPMix
# torch.autograd.set_detect_anomaly(True)

def prefix(text_list):
    prefix_text_list = []
    prefix_target_list = []
    for text in text_list:
        split_text = text.strip().split(" ")
        slice_id = random.randint(0, len(split_text) - 1)
        prefix_text = split_text[:slice_id]
        prefix_target = split_text[slice_id:]
        prefix_text_list.append(" ".join(prefix_text))
        prefix_target_list.append(" ".join(prefix_target))
    return prefix_text_list, prefix_target_list

def train(model, data_loader, r_data_loader,optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, do_amp=False,
          do_two_optim=False,tpmix=None):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if do_two_optim:
        metric_logger.add_meter('lr1', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('lr2', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    else:
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita_mixed', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_patch_pred', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('recall', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_prefix', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)
        r_data_loader.sampler.set_epoch(epoch)
    patch_pred_recall = 0.0
    patch_pred_acc = 0.0
    # if epoch>=1:
    #     do_mixup=True  
    # else:
    #     do_mixup=False
    if tpmix is not None:
        do_mixup=True  
    else:
        do_mixup=False
    subarea_iter = iter(r_data_loader)
    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if epoch > 0:
            alpha = config['alpha']
            gamma = 1.0
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader)) 
            gamma = 1.0 * min(1, i / len(data_loader)) 
        flag_region = random.random() < config['regions']['iter_perc']
        
        if flag_region:
            try:
                region_batch = next(subarea_iter)
            except StopIteration:
                subarea_iter = iter(r_data_loader)
                region_batch = next(subarea_iter)
            
            image_region, region_batch = region_batch[0].to(device, non_blocking=True), [
                t.to(device) if t is not None else None for t in region_batch[1:]]

            idx_to_group_img, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, \
                image_atts, target_bbox, is_image = region_batch
            optimizer.zero_grad()

            loss_patch_pred ,acc, recall = model(image = image_region, 
                                alpha = gamma,
                                is_region_forward = True, 
                                text_ids = text_ids, 
                                text_atts = text_atts,
                                image_atts = image_atts,
                                idx_to_group_img = idx_to_group_img 
                                )
            
            if acc>patch_pred_acc and recall > patch_pred_recall:
                patch_pred_acc = acc
                patch_pred_recall = recall
            
            # gamma = (patch_pred_acc + patch_pred_recall)*0.5 + 0.2 - math.fabs(patch_pred_acc - patch_pred_recall)*0.5
            if do_amp:
                from apex import amp
                with amp.scale_loss(loss_patch_pred, optimizer) as scaled_loss:
                    # logger.info('scaled loss: {}'.format(str(scaled_loss)))
                    scaled_loss.backward()
            else:
                patch_pred_loss.backward()
            if do_amp:
                nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.)

            optimizer.step()
            
        

        optimizer.zero_grad()
        image = image.to(device, non_blocking=True)
        
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=20, return_tensors="pt").to(device)
        prefix_text_list, prefix_target_list = prefix(text)
        prefix_target_list = [each + config['eos'] for each in prefix_target_list]
        prefix_input = tokenizer(prefix_text_list, padding='longest', truncation=True, max_length=20,
                                 return_tensors="pt").to(device)
        prefix_target = tokenizer(prefix_target_list, padding='longest', truncation=True, max_length=20,
                                  return_tensors="pt").to(device)
        #需要模型输出原始的图像token和文本token分类权重
        loss_mlm, loss_ita, loss_itm, loss_prefix, token_att, image_tokens, un_mixed_prediction_distribution = model(image, text_input, alpha=alpha, prefix_input=prefix_input,
                                                          prefix_target=prefix_target,is_mix_up=False)
        loss = loss_mlm + loss_ita + loss_itm + loss_prefix
        
        if do_amp:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                # logger.info('scaled loss: {}'.format(str(scaled_loss)))
                scaled_loss.backward()
        else:
            loss.backward()
        if do_amp:
            nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.)
        optimizer.step()

        if do_mixup:
            optimizer.zero_grad()
            target = torch.tensor([i for i in range(config['batch_size'])]).to(image.device) 
            with torch.no_grad():       
                image_mixed, mixed_targeted, mixed_prediction_distribution = tpmix(x= image, target=target, att= token_att, un_mixed_prediction_distribution=un_mixed_prediction_distribution)
            loss_tpmix = model(image_mixed, text = text_input, is_mix_up=True, label_mixed = mixed_prediction_distribution)
            
            loss = loss_tpmix * gamma
            if do_amp:
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    # logger.info('scaled loss: {}'.format(str(scaled_loss)))
                    scaled_loss.backward()
            else:
                loss.backward()
            if do_amp:
                nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.)
            optimizer.step()

        if flag_region:
            metric_logger.update(loss_patch_pred=loss_patch_pred.item())
            metric_logger.update(acc=acc)
            metric_logger.update(recall=recall)
        else:
            metric_logger.update(loss_patch_pred=-1.0)
            metric_logger.update(acc=0)
            metric_logger.update(recall=0)
        if do_mixup:
            metric_logger.update(loss_ita_mixed=loss_tpmix.item())
        else:
            metric_logger.update(loss_ita_mixed=0)

        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_prefix=loss_prefix.item())
        if do_two_optim:
            metric_logger.update(lr1=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr2=optimizer.param_groups[2]["lr"])
        else:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

            # gather the stats from all processes
        del image, loss, scaled_loss, loss_mlm, loss_ita, loss_itm, loss_prefix, text_input, prefix_target, prefix_input
        if flag_region:
            del loss_patch_pred
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(
            device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask)
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    image_feats = []
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder.visual(image, skip_last_layer=True)
        # image_feat = model.visn_layer_norm(model.visn_fc(image_feat))
        # image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)
        image_feats.append(image_feat)
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start + i].repeat(config['k_test'], 1, 1)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        _, output = model.fusion_encoder.bert(encoder_embeds=text_feats[topk_idx],
                                         attention_mask=text_atts[topk_idx],
                                         encoder_hidden_states=encoder_output,
                                         encoder_attention_mask=encoder_att,
                                         return_dict=False,
                                        )
        score = model.itm_head(output[:, 0, :])[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score.float()

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        _, output = model.fusion_encoder.bert(encoder_embeds=text_feats[start + i].repeat(config['k_test'], 1, 1),
                                         attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
                                         encoder_hidden_states=encoder_output,
                                         encoder_attention_mask=encoder_att,
                                         return_dict=False,
                                        )
        score = model.itm_head(output[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score.float()

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result
def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset ####
    print("Creating dataset")
    datasets = create_dataset('pretrain_with_test', config)
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, True, True], num_tasks, global_rank)
    else:
        samplers = [None,None,None]
         
    _, r_dataset,_ = datasets
    data_loader, r_data_loader, test_loader = \
        create_loader(datasets, samplers, batch_size=[config['batch_size'], config['regions']['batch_size'], config['batch_size']], num_workers=[15,8, 8 ], is_trains=[True,True,True],
                      collate_fns=[None, r_dataset.collate_fn, None])
   
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = MPLUG(config=config, tokenizer=tokenizer)
    
    model = model.to(device)
    
    if not args.do_two_optim:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
    else:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_two_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.do_amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['module']
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            
        else:
            if "clip_name" not in config:
                num_patches = int(config["image_res"] * config["image_res"] / (16 * 16))
            else:
                num_patches = int(config["image_res"] * config["image_res"] / (14 * 14))
            print('num_pathes:',num_patches)
            pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

            pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                         pos_embed.unsqueeze(0))
            state_dict['visual_encoder.visual.positional_embedding'] = pos_embed
            if config['distill']:
                pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

                pos_embed = resize_pos_embed(state_dict['visual_encoder_m.visual.positional_embedding'].unsqueeze(0),
                                             pos_embed.unsqueeze(0))
                state_dict['visual_encoder_m.visual.positional_embedding'] = pos_embed

        model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)

    model_without_ddp = model
    if args.distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        import apex
        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        model_without_ddp = model.module
    tpmix = TPMix( label_smoothing=args.smoothing, 
            num_classes=config['batch_size'], min_side_ratio=config['min_side_ratio'],max_side_ratio=config['max_side_ratio'], side=config['image_res']//config['patch_size'], patch_size = config['patch_size'] )
    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)
        # score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)
        # test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img,
        #                            test_loader.dataset.img2txt)
       
        if epoch>=20:
            tpmix = None
        train_stats = train(model, data_loader,r_data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                            config, do_amp=args.do_amp, do_two_optim=args.do_two_optim , tpmix = tpmix)
        # score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)
        if utils.is_main_process():
            # test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img,
            #                        test_loader.dataset.img2txt)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            # test_log_stats = {**{f'test_{k}': v for k, v in test_result.items()},
            #              'epoch': epoch,
            #              }
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))


            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
            # with open(os.path.join(args.output_dir, "itm_eval.txt"), "a") as f:
            #     f.write(json.dumps(test_log_stats) + "\n")
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=True, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dec_layers', default=12, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--do_two_optim', action='store_true')
    parser.add_argument('--do_amp', action='store_true')

    parser.add_argument('--smoothing', type=float, default=0.1, 
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--min_side_ratio', type=float, default=0.4,
                        help='the minimum side ration of crop rectangle')
    parser.add_argument('--max_side_ratio', type=float, default=0.6,
                        help='the maximum side ration of crop rectangle')

    args = parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config['dec_layers'] = args.dec_layers
    config['text_encoder'] = args.text_encoder
    config['text_decoder'] = args.text_decoder
    config['smoothing'] = args.smoothing
 
    
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
