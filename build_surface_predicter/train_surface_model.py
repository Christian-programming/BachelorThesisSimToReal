from taskonomy_network import TaskonomyEncoder, TaskonomyDecoder, TaskonomyNetwork,  TASKS_TO_CHANNELS
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2
import numpy as np
import sys
import torch
import time
from collections import deque
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
from replay_buffer_depth import ReplayBufferDepth


def time_format(sec):
    """
    Args:
    param1():
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs, 2)


def main(args):
    t0 = time.time()
    TASKONOMY_PRETRAINED_WEIGHT_FILES = ["normal_decoder-8f18bfb30ee733039f05ed4a65b4db6f7cc1f8a4b9adb4806838e2bf88e020ec.pth", "normal_encoder-f5e2c7737e4948e3b2a822f584892c342eaabbe66661576ba50db7cdd40561c5.pth"]
    path_de = 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/' + str(TASKONOMY_PRETRAINED_WEIGHT_FILES[0])
    path_en = 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/' + str(TASKONOMY_PRETRAINED_WEIGHT_FILES[1])
    model = TaskonomyNetwork(load_encoder_path=path_en, load_decoder_path=path_de)
    model_path = "trained_real_world_model"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save_model(model_path + "/real_world_model-{}".format(0))
    model.encoder.eval_only = False
    model.decoder.eval_only = False

    for param in model.parameters():
        param.requires_grad = True

    model.cuda()
    model.train()

    lr = 3e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    pathname = "real_world_model_surface_normals"
    pathname += dt_string
    tensorboard_name = 'runs/' + pathname
    writer = SummaryWriter(tensorboard_name)

    size = 256
    memory = ReplayBufferDepth((size, size), (size, size, 3), (size, size, 3), 15001, "cuda")

    print("Load buffer ...")
    memory.load_memory_normals(args.path_buffer)
    print("... buffer size {} loaded".format(memory.idx))

    torch.cuda.empty_cache()
    batch_size = 32
    scores_window = deque(maxlen=100)
    epochs = int(100e4)
    index = 14
    array_depth = memory.depth[index]
    array_normals = memory.normals[index]
    obs = memory.obses[index]
    frame = cv2.imwrite("surface_image_buffer{}.png".format(index), np.array(array_normals * 255))
    frame = cv2.imwrite("depth_image_buffer{}.png".format(index), np.array(array_depth * 255))
    frame = cv2.imwrite("image_buffer{}.png".format(index), np.array(obs))
    index = 11042
    array_depth = memory.depth[index]
    array_normals = memory.normals[index]
    obs = memory.obses[index]
    frame = cv2.imwrite("surface_image_buffer{}.png".format(index), np.array(array_normals * 255))
    frame = cv2.imwrite("depth_image_buffer{}.png".format(index), np.array(array_depth * 255))
    frame = cv2.imwrite("image_buffer{}.png".format(index), np.array(obs))

    for epoch in range(epochs):
        print('\rEpisode {}'.format(epoch), end="")
        rgb_batch, depth_batch, normal_batch = memory.sample(batch_size)
        x_recon = model(rgb_batch.cuda())

        optimizer.zero_grad()
        loss = F.mse_loss(x_recon, normal_batch.cuda())
        loss.backward()
        optimizer.step()
        scores_window.append(loss.item())
        mean_loss = np.mean(scores_window)
        writer.add_scalar('loss', loss.item(), epoch)
        writer.add_scalar('mean_loss', mean_loss, epoch)

        if epoch % 10 == 0 and epoch > 1000:
            text = "Epochs {}  loss {:.5f}  ave loss {:.5f}  time {}  \r".format(epoch, loss, mean_loss, time_format(time.time() - t0))
            print("  ")
            print(text)
            for index in [14, 11042]:
                RGB_state = memory.obses[index]
                RGB_state = TF.to_tensor(RGB_state).cuda()
                RGB_state = RGB_state.unsqueeze(0)
                print("shape", RGB_state.shape)
                model.eval()
                outPut = model(RGB_state).detach().cpu().numpy()
                model.train()
                outPut = np.array(outPut.squeeze(0))
                print(outPut.shape)
                outPut = outPut.transpose(1, 2, 0)
                print("max value ", np.max(outPut))
                outPut = outPut * 255
                print(np.max(outPut))
                cv2.imwrite("Surface_normals_prediction_index{}_epoch_{}.png".format(index, epoch), np.array(outPut))
            model.save_model(model_path + "/model_step_{}_eval_loss_{:.10f}".format(epoch, loss.item()))
        continue

        if epoch % 75 == 0:
            model.eval()
            eval_loss = 0
            evaL_size = 25
        for i in range(evaL_size):
            rgb_batch, depth_batch, normal_batch = memory_valid.sample(batch_size)
            x_recon = model(rgb_batch.cuda()).detach()
            loss = F.mse_loss(x_recon, normal_batch.cuda())
            eval_loss += loss
        eval_loss = eval_loss / evaL_size
        model.save_model(model_path + "/model_step_{}_eval_loss_{:.10f}".format(epoch, eval_loss))
        model.train()
        text = "Eval model {} eval loss {:10f} time {}  \r".format(epoch, eval_loss, time_format(time.time() - t0))
        writer.add_scalar('eval_loss', eval_loss, epoch)
        print("  ")
        print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_buffer', type=str, required=True, help="path train replay buffer")
    arg = parser.parse_args()
    main(arg)
