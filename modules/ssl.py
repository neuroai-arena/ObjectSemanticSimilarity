import functorch
import torch
import torchvision
from torchvision import transforms

from modules.loss_module import LossModule
from networks.heads import MLPHead
from utils.augmentations import get_transformations, get_resized_crop, get_flip, get_jitter, get_grayscale, \
    get_transform_list
from utils.constants import LOSS, SIMILARITY_FUNCTIONS, SIMILARITY_FUNCTIONS_SIMPLE, DATASETS
from utils.general import is_target_needed, run_forward
from utils.losses import BYOL


class Ssl(LossModule):

    def __init__(self,args, fabric, net=None, net_target=None, **kwargs):
        self.args = args
        self.fabric=fabric
        net.add_module("projector", MLPHead(args,net.num_output, args.hidden_dim, args.feature_dim, args.hidden_layers))
        net.register_buffer("proj_output", torch.empty((2 * args.batch_size, args.feature_dim)), persistent=False)

        if is_target_needed(args):
            net.register_buffer("prediction", torch.empty((2 * args.batch_size, args.feature_dim)), persistent=False)
            net.add_module("predictor",MLPHead(args,args.feature_dim, args.hidden_dim, args.feature_dim, args.hidden_layers))

        self.loss = LOSS[args.main_loss](args, SIMILARITY_FUNCTIONS[args.similarity], fabric)
        self.k=0
        self.loss_store = self.fabric.to_device(torch.zeros((1,)))
        self.loss_cpt = 1e-5

    @classmethod
    def get_args(cls, parser):
        parser.add_argument('--lambda_vicreg', default=25, type=float)
        parser.add_argument('--mu_vicreg', default=25, type=float)
        parser.add_argument('--v_vicreg', default=1, type=float)
        parser.add_argument('--temperature', default=0.1, type=float)
        parser.add_argument('--classic_weight', default=1, type=float)

        parser.add_argument('--feature_dim', default=512, type=int)
        parser.add_argument('--hidden_dim', default=1024, type=int)
        parser.add_argument('--hidden_layers', default=2, type=int)

        return parser

    @torch.no_grad()
    def eval(self, net, *args):
        dict = {"ssl_loss": self.loss_store.item()/self.loss_cpt}
        self.loss_store[:]=0
        self.loss_cpt = 1e-5
        return dict

    def apply(self, net, rep=None, rep_target=None, net_target=None, data=None, **kwargs):
        # batch_data["projection"] = self.net.projector(batch_data["representation"])
        # print(data[0][0][0].shape)
        # if self.k < 100:
        #     torchvision.transforms.functional.to_pil_image(((data[0][0][10]*0.5 +0.5)*255).to(torch.uint8)).save("../gym_results/test_images/vis/"+str(self.k)+"_1.png")
        #     torchvision.transforms.functional.to_pil_image(((data[0][1][10]*0.5 +0.5)*255).to(torch.uint8)).save("../gym_results/test_images/vis/"+str(self.k)+"_2.png")
        #     torchvision.transforms.functional.to_pil_image(((data[0][0][10])*255).to(torch.uint8)).save("../gym_results/test_images/vis/"+str(self.k)+"_1.png")
        #     torchvision.transforms.functional.to_pil_image(((data[0][1][10])*255).to(torch.uint8)).save("../gym_results/test_images/vis/"+str(self.k)+"_2.png")
        #     self.k+=1
        # net.proj_output = net.projector(rep)
        net.proj_output = run_forward(self.args, rep, net.projector)

        if self.args.main_loss in ['BYOL']:
            net.prediction = net.predictor(net.proj_output)
            net_target.proj_output = run_forward(self.args, rep, net_target.projector).detach()
            # net_target.proj_output = net_target.projector(rep_target).detach()
            y1, y2 = BYOL.get_byol_output(net_target.proj_output, net.prediction)
            loss = self.loss(y1, y2)
        else:
            y1, y2 = net.proj_output.split(net.proj_output.shape[0]//2)
            loss = self.loss(y1, y2)

        loss_mean = loss.mean()
        self.loss_store = self.loss_store + loss_mean.detach()
        self.loss_cpt += 1
        return self.args.classic_weight*loss_mean


class DecoupledSsl(LossModule):
    def __init__(self, args, fabric, net=None, net_target=None, **kwargs):
        # super().__init__(args, net, net_target, **kwargs)
        self.args=args
        net.add_module("dec_projector", MLPHead(args,net.num_output, args.hidden_dim, args.feature_dim, args.hidden_layers))
        self.loss = LOSS[args.main_loss](args, SIMILARITY_FUNCTIONS[args.similarity], fabric)

        # class ToFloat:
        #     def __call__(self, imgs):
        #         return imgs.to(torch.float32)
        #
        # rgb_mean, rgb_std = DATASETS[args.dataset]['rgb_mean'], DATASETS[args.dataset]['rgb_std']
        # normalize = transforms.Normalize(mean=rgb_mean, std=rgb_std)
        # transformations = []
        # if args.image_padding:
        #     transformations.append(transforms.Pad(args.image_padding))
        # if args.flip:
        #     transformations.append(get_flip(args))
        # if args.jitter != 0 and not args.unijit:
        #     transformations.append(get_jitter(args))
        # if args.grayscale and not args.unijit:
        #     transformations.append(get_grayscale(args))
        # if args.blur:
        #     transformations.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=args.blur)], p=0.2))
        # transformations.append(ToFloat())
        # transformations.append(normalize)
        # aug = transforms.Compose(transformations)
        # self.train_transform = transforms.Lambda(lambda x: torch.stack([aug(x_) for x_ in x]))
        # self.cpt = 0

    @classmethod
    def get_args(cls, parser):
        return parser

    def apply(self, net, net_target=None, data=None, **kwargs):
        # batch_data["projection"] = self.net.projector(batch_data["representation"])
        # x_aug = self.train_transform(torch.cat((data[4],data[5]),dim=0).to(self.args.device)
        y1, y2 = net.proj_output.split(net.proj_output.shape[0]//2)
        # x_aug = self.train_transform(data[4]).to(self.args.device)
        # x_aug = self.train_transform(data[4]).to(self.args.device)
        # x_aug = data[4].to(self.args.device).to(torch.float32)
        x_aug = data[5]
        y3 = net.dec_projector(net(x_aug))
        # y3_image = ((x_aug*0.5 +0.5)*255).to(torch.uint8)
        # y2_image = ((data[0][1]*0.5 +0.5)*255).to(torch.uint8)
        # y1_image = ((data[0][0]*0.5 +0.5)*255).to(torch.uint8)
        # for i in range(3):
        #     torchvision.transforms.functional.to_pil_image(y3_image[i].squeeze()).save("../gym_results/test_images/trans"+str(i)+".png")
        #     torchvision.transforms.functional.to_pil_image(y2_image[i].squeeze()).save("../gym_results/test_images/trans"+str(i)+"_1.png")
        #     torchvision.transforms.functional.to_pil_image(y1_image[i].squeeze()).save("../gym_results/test_images/trans"+str(i)+"_2.png")
        loss = self.args.multiscale_weight*self.loss(y3, y1)
        return loss.mean()

class StandSsl(LossModule):

    def __init__(self, args, fabric, net=None, net_target=None, **kwargs):
        # super().__init__(args, fabric, net=net, net_target=net_target, **kwargs)
        self.args = args
        self.fabric=fabric
        img_size = DATASETS[args.dataset]['img_size'] if "img_size" in DATASETS[args.dataset] else None
        self.train_transform  = get_transform_list(args, tensor_normalize=False, crop_size=img_size)
        # self.train_transform = transforms.Lambda(lambda x: torch.stack([aug(x_) for x_ in x]))

        net.add_module("standard_projector", MLPHead(args,net.num_output, args.hidden_dim, args.feature_dim, args.hidden_layers))
        net.register_buffer("standard_proj_output", torch.empty((args.batch_size, args.feature_dim)), persistent=False)
        mean, std = DATASETS[args.dataset]['rgb_mean'], DATASETS[args.dataset]['rgb_std']
        self.loss = LOSS[args.main_loss](args, SIMILARITY_FUNCTIONS[args.similarity], fabric, temperature=args.standard_temperature)

        assert mean[0] == 0.5 and std[0] == 0.5, "Only work with normalization in -1, 1"
        # self.cpt = 0

    @classmethod
    def get_args(cls,parser):
        parser.add_argument('--standard_temperature', default=0.1, type=float)
        return parser

    def apply(self, net, net_target=None, rep=None, data=None, **kwargs):
        e1 = rep[:rep.shape[0]//2]

        x = data[0][0]
        x_aug = (self.train_transform((x+1)/2) - 0.5)/0.5
        # net_fc = functorch.functionalize(net.forward)
        e2 = net(x_aug)
        # e2 = net((x+1)/2)
        #
        y = net.projector(torch.cat((e1, e2), dim=0))
        y1, y2 = y.split(y.shape[0] // 2)
        # loss = net.standard_projector(e1).mean()
        loss = self.loss(y1, y2)
        # y3_image = ((x_aug*0.5 +0.5)*255).to(torch.uint8)
        # y2_image = ((data[0][1]*0.5 +0.5)*255).to(torch.uint8)
        # y1_image = ((data[0][0]*0.5 +0.5)*255).to(torch.uint8)
        # for i in range(3):
        #     torchvision.transforms.functional.to_pil_image(y3_image[i].squeeze()).save("../gym_results/test_images/trans"+str(i)+".png")
        #     torchvision.transforms.functional.to_pil_image(y2_image[i].squeeze()).save("../gym_results/test_images/trans"+str(i)+"_1.png")
        #     torchvision.transforms.functional.to_pil_image(y1_image[i].squeeze()).save("../gym_results/test_images/trans"+str(i)+"_2.png")
        return loss.mean()

    @torch.no_grad()
    def eval(self, net, *args):
        return {}


