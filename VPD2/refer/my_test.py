import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

import transforms as T
import utils
from transformers.models.clip.modeling_clip import CLIPTextModel
from models_refer.model import VPDRefer
import numpy as np
from PIL import Image
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms


import sys
sys.path.append('../stablediffusion')
sys.path.append('../')
sys.path.append('./')



def get_dataset(image_set, transform, args):
    from data.dataset_refer_clip import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes


def evaluate(model, data_loader, clip_model, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'

    with torch.no_grad():
        i = 0
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()
            for idx in range(sentences.size(-1)):
                
                embedding = clip_model(input_ids=sentences[:, :, idx]).last_hidden_state
                attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
                output = model(image, embedding)
                output = output.cpu()
                output_mask = output.argmax(1).data.numpy()

                #Part I added to the code

                #image

                plt.imsave(f"input_image{i}.png",image.cpu())
                
                #predicted_mask

                output_mask_np = np.tile(output_mask, (3, 1, 1))
                image_np = image.cpu().numpy()

                reshaped_output_mask = np.transpose(output_mask_np, (1, 2, 0))
                reshaped_image_np = np.transpose(image_np[0], (1, 2, 0))

                masked_image = reshaped_image_np.copy()
                masked_image = np.where(reshaped_output_mask.astype(int),
                          np.array([255,255,0], dtype='uint8'),
                          masked_image)
                masked_image = masked_image.astype(np.uint8)

                result = cv2.addWeighted((reshaped_image_np * 255).astype(np.uint8), 1.0, masked_image, 0.7, 0)
                          
                cv2.imwrite(f"predidcted_image_{i}.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

                #target_mask

                target_np = np.tile(target, (3, 1, 1))
                reshaped_target_np = np.transpose(target_np, (1, 2, 0))

                masked_image = reshaped_image_np.copy()
                masked_image = np.where(reshaped_target_np.astype(int),
                          np.array([255,255,0], dtype='uint8'),
                          masked_image)
                masked_image = masked_image.astype(np.uint8)

                result = cv2.addWeighted((reshaped_image_np * 255).astype(np.uint8), 1.0, masked_image, 0.7, 0)
                          
                cv2.imwrite("/content/" + f"target_image_{i}.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

                i = i + 1

                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def main(args):
    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args) 
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)    
    single_model = VPDRefer(sd_path='checkpoints/v1-5-pruned-emaonly.ckpt',
                      neck_dim=[320,640+args.token_length,1280+args.token_length,1280]
                      )

    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'], strict=False)
    model = single_model.to(device)

    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.cuda()
    clip_model = clip_model.eval()

    evaluate(model, data_loader_test, clip_model, device=device)

if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
