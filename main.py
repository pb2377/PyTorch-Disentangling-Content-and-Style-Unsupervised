import os
import torch
import models
import utils
from torch.utils import data
import torch.optim as optim
from torchvision import transforms
import shutil
from PIL import Image
import vgg
from time import process_time

MODE = 'train'
# MODE = 'eval'
model_type = 'hourglass'
hourglass = True
cuda_id = [4]
if len(cuda_id) > 1:
    data_parallel = True
else:
    data_parallel = False
    cuda_id = cuda_id[0]


def main(mode):
    # Dataloaders
    batch_size = 8
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 4}

    # Model
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.GeomVAE()

    batch_norm = True
    vgg_extractor = models.ExtractorVGG(features=vgg.make_layers(vgg.cfgs['E'], batch_norm=batch_norm),
                                        arch='vgg19', batch_norm=batch_norm)
    vgg_extractor.eval()

    if data_parallel and torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=cuda_id).cuda(cuda_id[0])
        vgg_extractor = torch.nn.DataParallel(vgg_extractor,  device_ids=cuda_id).cuda(cuda_id[0])
    elif torch.cuda.is_available():
        model = model.cuda(cuda_id)
        vgg_extractor.cuda(cuda_id)
    print('Training {} Type Model'.format(model_type.upper()))
    # if os.path.exists('model-weights.out'):
    #     model.load_state_dict(torch.load('model-weights.pt',
    #                                      map_location='cuda' if torch.cuda.is_available() else 'cpu'))

    if mode.lower() == 'train':
        dataset = utils.Cats()
        dataloader = data.DataLoader(dataset, **params)
        # print(len(dataloader))

        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        print("   Training {} parameters "
              "of {} total parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad),
                                              sum(p.numel() for p in model.parameters())))

        # print(params_to_update)
        # optimizer = optim.Adam(params_to_update, lr=1e-4, betas=(0.5, 0.999), weight_decay=5e-5)
        optimizer = optim.Adam(params_to_update, lr=1e-4, betas=(0.5, 0.999))
        max_eps = 160
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=round(max_eps / 3), gamma=0.1)
        # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        model.train()
        for epoch in range(0, max_eps):
            running_losses = [0., 0., 0.]
            first_batch = True
            t0 = process_time()
            for image_batch, _ in dataloader:
                if data_parallel and torch.cuda.is_available():
                    image_batch = image_batch.cuda(cuda_id[0])
                elif torch.cuda.is_available():
                    image_batch = image_batch.cuda(cuda_id)

                # zero_grad
                optimizer.zero_grad()

                # zero_grad
                x, y, losses = model(image_batch)
                rc_loss, prior_loss, kl_loss = losses
                rc_loss += vgg_extractor(image_batch, x)
                losses = (rc_loss, prior_loss, kl_loss)
                if data_parallel and torch.cuda.is_available():
                    losses = [i.mean() for i in losses]
                # loss = rc_loss + prior_loss + kl_loss
                # print([round(i.item(), 4) for i in losses])
                # loss = losses[2]
                loss = sum(losses)
                # loss = rc_loss
                loss.backward()
                # utils.plot_grad_flow(model.named_parameters())
                # raise NotImplementedError
                optimizer.step()
                running_losses = [running_losses[i] + losses[i].item() for i in range(len(losses))]
                if first_batch:
                    # combine images into one
                    # [input, normalize(y), output]
                    outdir = 'outputs-17'
                    if not os.path.exists('{}/{}/train_viz'.format(outdir, model_type)):
                        os.makedirs('{}/{}/train_viz'.format(outdir, model_type))
                    transf = transforms.ToPILImage()
                    x_in = image_batch[0, :, :, :]
                    y_image = assemble_landmarks(x_in, y[0, :, :, :].sum(0, keepdim=True))
                    x_in = transf(x_in.cpu()).convert('RGB')
                    x_out = transf(x[0, :, :, :].cpu()).convert('RGB')
                    # print(x[0, :, :, :].min(), x[0, :, :, :].max(), x[0, :, :, :].mean())
                    image_out = Image.new(mode='RGB', size=(256*3, 256))
                    x_offset = 0
                    for im in [x_in, y_image, x_out]:
                        image_out.paste(im, (x_offset, 0))
                        x_offset += 256
                    image_out.save('{}/{}/train_viz/epoch-{}.jpg'.format(outdir, model_type, epoch))
                    first_batch = False
                    del x_in, y_image, image_out, x_out

            total_loss = sum(running_losses) / len(dataloader)
            rc_loss = running_losses[0] / len(dataloader)
            prior_loss = running_losses[1] / len(dataloader)
            kl_loss = running_losses[2] / len(dataloader)
            t1 = process_time()
            esti = (max_eps - epoch+1) * ((t1-t0)/3600)
            print("Epoch {}/{} - Total Loss {:.3f} - Reconstruction Loss {:.3f} - "
                  "Prior Loss {:.3f} - KL Loss {:.3f} - {} Hours Remaining ".format(epoch + 1, max_eps, total_loss,
                                                                                    rc_loss, prior_loss, kl_loss,
                                                                                    round(esti)))
            lr_scheduler.step()
            if not epoch % 10:
                export_model(model.state_dict(), '{}-weights.pt'.format(model_type))
                export_model(model, '{}-full-model.pt'.format(model_type))
                print("Model Saved!")

        export_model(model.state_dict(), '{}-weights.pt'.format(model_type))
        export_model(model, '{}-full-model.pt'.format(model_type))

    elif mode.lower() == 'eval':
        # model.load_state_dict(torch.load('model-out.pt'))
        # model.load_state_dict(torch.load('model-weights.pt', map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        model = torch.load('full-model.pt', map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()

        print("WARNING: train=False dataloader currently using Train list ")
        dataset = utils.Cats(train=False)
        dataloader = data.DataLoader(dataset, **params)
        if os.path.exists('outputs'):
            shutil.rmtree('outputs', ignore_errors=False, onerror=None)
        for image_batch, image_paths in dataloader:
            if torch.cuda.is_available():
                image_batch = image_batch.cuda()

            x, y, losses = model(image_batch)
            export_images(x, image_paths, out_dir='outputs/reconstructed')
            export_images(y.sum(1, keepdim=True), image_paths, out_dir='outputs/landmarks', normalize=True)
            raise NotImplementedError


def assemble_landmarks(image, y):
    transf = transforms.ToPILImage()
    y -= y.min()
    y /= y.max()
    y_pad = torch.zeros_like(image)
    y_pad[1, :, :] = y
    alpha = y

    image_out = 0.5*(1 - alpha)*image + alpha * y_pad
    image_out = transf(image_out.cpu()).convert('RGB')
    # image_out = Image.new("RGB", image.size)
    # image_out = Image.alpha_composite(image_out, image)
    # image_out = Image.alpha_composite(image_out, y)
    return image_out


def export_model(output_file, outpath):
    if os.path.exists(outpath):
        os.remove(outpath)
    torch.save(output_file, outpath)


def export_images(image_tensor, image_paths, out_dir, normalize=False):
    for i in range(image_tensor.size(0)):
        image = image_tensor[i, :, :, :]
        out_path = os.path.join(out_dir, image_paths[i])
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        save_image(image.cpu(), out_path, normalize=normalize)


def save_image(image, out_path, normalize=False):
    if normalize:
        image -= image.min()
        image /= image.max()
    transf = transforms.ToPILImage()
    image = transf(image).convert('RGB')
    image.save(out_path)


if __name__ == '__main__':
    main(MODE)
