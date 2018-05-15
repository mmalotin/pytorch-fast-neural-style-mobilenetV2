import cv2
import torch
from fnst_modules import TransformerMobileNet
from torchvision import transforms

MODEL = 'models/mosaic.pth'
IMAGE_SIZE = 300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255).unsqueeze(0).to(device))])


def postprocess(tens):
    img = tens.permute(1, 2, 0).clamp(0, 255)
    img = img.cpu().numpy()
    img = img.astype("uint8")
    return img


def prepare_net(f, net):
    state_dict = torch.load(f)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()


def main():
    with torch.no_grad():
        net = TransformerMobileNet()
        prepare_net(MODEL, net)

        capture = cv2.VideoCapture(0)

        while True:
            _, im = capture.read()
            im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
            t = transform(im)
            res = net(t)[0]
            im = postprocess(res)
            cv2.imshow('webcam', im)
            if cv2.waitKey(1) == 27:  # press Esc to end
                break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
