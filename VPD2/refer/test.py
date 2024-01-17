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
            #print(f"data{data}")
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()
            #print("SENTENCE")
            #print(sentences)
            #print(sentences.shape)
            #print(sentences.size(-1))
            for idx in range(sentences.size(-1)):
                
                embedding = clip_model(input_ids=sentences[:, :, idx]).last_hidden_state
                #print(embedding.shape)
                attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
                output = model(image, embedding)

                #print("HOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                #print(output.shape)
                #print(output)
                #print(f"sentences {sentences}")
                """if(i <300):
                  visualize_output(image, output, target,100+i)  # Add this function
                i = i+1"""


                output = output.cpu()
                output_mask = output.argmax(1).data.numpy()
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
    #print(results_str)

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

def visualize_output(image, output, target, index):
    image = transforms.ToPILImage()(image.squeeze().cpu())
    output = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()

    # Transpose the image tensor to (height, width, channels) format
    image = np.transpose(image, (0, 1, 2))
    target = np.transpose(target, (1, 2, 0))

    # Visualize the input image
    plt.figure(figsize=(15, 5))

    # Subplot 1: Input Image
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(f'input_image_{index}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Subplot 2: Superposed Target Mask on the Real Image (Class 0)
    plt.figure(figsize=(15, 5))
    plt.imshow(image, alpha=0.8)  # Adjust alpha for clarity
    plt.imshow(target[:, :, 0], alpha=0.5, cmap='plasma', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig(f'target_mask_class_0_{index}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Subplot 3: Superposed Predicted Mask on the Real Image (Class 1)
    plt.figure(figsize=(15, 5))
    plt.imshow(image, alpha=0.8)  # Adjust alpha for clarity
    plt.imshow(output[1], alpha=0.5, cmap='plasma', vmin=0, vmax=1, interpolation='nearest')
    plt.axis('off')
    plt.savefig(f'predicted_mask_class_1_{index}.png', bbox_inches='tight', pad_inches=0)
    plt.close()




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
    #print("HOOOOOOOOOOOO")
    device = torch.device(args.device)
    dataset_test, _ = get_dataset('train',get_transform(args=args), args) 
    #print("OKKKKKKKKKKK")
    #print(dataset_test)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    #print(args.model)
    
    single_model = VPDRefer(sd_path='checkpoints/v1-5-pruned-emaonly.ckpt',
                      neck_dim=[320,640+args.token_length,1280+args.token_length,1280]
                      )

    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'], strict=False)
    model = single_model.to(device)

    #print("OK22222")

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
