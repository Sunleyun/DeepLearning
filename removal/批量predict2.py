import argparse
from distutils.util import strtobool
import os

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from src.model import PConvUNet


def main(args,im_name):
    # Define the used device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Define the model
    print("Loading the Model...")
    model = PConvUNet(finetune=False, layer_size=7)
    model.load_state_dict(torch.load(args.model, map_location=device)['model'])
    model.to(device)
    model.eval()

    # Loading Input and Mask
    print("Loading the inputs...")
    org = Image.open(args.img)
    org = TF.to_tensor(org.convert('RGB'))
    mask = Image.open(args.mask)
    mask = TF.to_tensor(mask.convert('RGB'))
    inp = org * mask

    # Model prediction
    print("Model Prediction...")
    with torch.no_grad():
        inp_ = inp.unsqueeze(0).to(device)
        mask_ = mask.unsqueeze(0).to(device)
        if args.resize:
            org_size = inp_.shape[-2:]
            inp_ = F.interpolate(inp_, size=256)
            mask_ = F.interpolate(mask_, size=256)
        raw_out, _ = model(inp_, mask_)
    if args.resize:
        raw_out = F.interpolate(raw_out, size=org_size)

    # Post process
    raw_out = raw_out.to(torch.device('cpu')).squeeze()
    raw_out = raw_out.clamp(0.0, 1.0)
    out = mask * inp + (1 - mask) * raw_out

    # Saving an output image
    print("Saving the output...")
    out = TF.to_pil_image(out)
    img_name = args.img.split('/')[-1]

    print(img_name)
    out.save(os.path.join("C:/Users/Adimi/Desktop/灰度金属图实验/hl_result2/val", "out_{}".format(im_name)))


if __name__ == "__main__":
    dir = 'D:/dataset/pix2pix_style/valA'
    imgList = os.listdir('D:/dataset/pix2pix_style/valA')
    dir2 = 'C:/Users/Adimi/Desktop/灰度金属图实验/seg_result/trans_val'#这是mask路径

    for count in range(0, len(imgList)):
        im_name = imgList[count]
        im_path = os.path.join(dir, im_name)
        print(im_path)
        newName = im_name[:-4]#仅保留文件名，去除后缀
        #注意，原图是jpg，mask是png


        parser = argparse.ArgumentParser(description="Specify the inputs")
        parser.add_argument('--img', type=str, default=im_path)
        parser.add_argument('--mask', type=str, default=dir2 +"/"+ newName+".png")
        #parser.add_argument('--model', type=str, default="pretrained_pconv.pth")
        parser.add_argument('--model', type=str, default="D:\code\partialconv-master1_traintest\ckpt/best4\models/160.pth")
        parser.add_argument('--resize', type=strtobool, default=False)
        parser.add_argument('--gpu_id', type=int, default=0)
        args = parser.parse_args()

        main(args,im_name)
