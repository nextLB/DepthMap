

# python run.py --encoder vitl --img-path assets/examples --outdir depth_vis


import sys
sys.path.append('/home/next_lb/桌面/next/Depth_Map')
from torchvision.transforms import Compose
from Depth.depth_anything.dpt import DepthAnything
from Depth.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import cv2
import torch
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


def detection_depth():
    imagePath = '/home/next_lb/桌面/next/depth_test_images/'
    outDir = './output'
    encoder = 'vitl'
    predOnly = None
    grayscale = None

    marginWidth = 50
    captionHeight = 60
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontThickness = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    depthAnything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()
    totalParams = sum(param.numel() for param in depthAnything.parameters())
    print('Total parameters: {:.2f}M'.format(totalParams / 1e6))
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    if os.path.isfile(imagePath):
        if imagePath.endswith('txt'):
            with open(imagePath, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [imagePath]
    else:
        filenames = os.listdir(imagePath)
        filenames = [os.path.join(imagePath, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()

    os.makedirs(outDir, exist_ok=True)

    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        h, w = image.shape[:2]

        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            depth = depthAnything(image)

        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.cpu().numpy().astype(np.uint8)

        if grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        filename = os.path.basename(filename)

        if predOnly:
            cv2.imwrite(os.path.join(outDir, filename[:filename.rfind('.')] + '_depth.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], marginWidth, 3), dtype=np.uint8) * 255
            combined_results = cv2.hconcat([raw_image, split_region, depth])

            caption_space = np.ones((captionHeight, combined_results.shape[1], 3), dtype=np.uint8) * 255
            captions = ['Raw image', 'Depth Anything']
            segment_width = w + marginWidth

            for i, caption in enumerate(captions):
                # Calculate text size
                text_size = cv2.getTextSize(caption, font, fontScale, fontThickness)[0]

                # Calculate x-coordinate to center the text
                text_x = int((segment_width * i) + (w - text_size[0]) / 2)

                # Add text caption
                cv2.putText(caption_space, caption, (text_x, 40), font, fontScale, (0, 0, 0), fontThickness)

            final_result = cv2.vconcat([caption_space, combined_results])

            cv2.imwrite(os.path.join(outDir, filename[:filename.rfind('.')] + '_img_depth.png'), final_result)




def main():
    detection_depth()


if __name__ == '__main__':
    main()




