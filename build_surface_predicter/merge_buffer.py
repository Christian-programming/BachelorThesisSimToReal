# Copyright 2021
# Author: Christian Leininger <info2016frei@gmail.com>

import argparse
from replay_buffer_depth import ReplayBufferDepth


def main(args):
    """  Takes the replay buffer with only sim data and real data
         creates a new buffer with a mix

    """
    size = 256
    memory_real = ReplayBufferDepth(
            (size, size), (size, size, 3), (size, size, 3), int(args.buffer_size_1), args.device)
    memory_sim = ReplayBufferDepth(
            (size, size), (size, size, 3), (size, size, 3), int(args.buffer_size_2), args.device)
    print("load buffer ..")
    memory_real.load_memory_normals(args.path1)
    print("real size", memory_real.idx)
    memory_sim.load_memory_normals(args.path2)
    print("sim size", memory_sim.idx)
    memory_simreal = ReplayBufferDepth(
            (size, size), (size, size, 3), (size, size, 3), int(args.buffer_size_new), args.device)
    for i in range(memory_real.idx):
        print(memory_simreal.idx)
        memory_simreal.depth[memory_simreal.idx] = memory_real.depth[i]
        memory_simreal.normals[memory_simreal.idx] = memory_real.normals[i]
        memory_simreal.obses[memory_simreal.idx] = memory_real.obses[i]
        memory_simreal.idx += 1
    print("size sim_real buffer ", memory_simreal.idx)
    for i in range(memory_sim.idx):
        print("new buffer indx {}  sim  index {}".format(memory_simreal.idx, i))
        memory_simreal.depth[memory_simreal.idx] = memory_sim.depth[i]
        memory_simreal.normals[memory_simreal.idx] = memory_sim.normals[i]
        memory_simreal.obses[memory_simreal.idx] = memory_sim.obses[i]
        memory_simreal.idx += 1
    print("size sim_real buffer ", memory_simreal.idx)
    memory_simreal.save_memory_normals(args.buffer_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=str, required=True, help="path first replay buffer")
    parser.add_argument('--buffer_size_1', type=int, required=True, help="size of first replay buffer")
    parser.add_argument('--path2', type=str, required=True, help="path sec replay buffer")
    parser.add_argument('--buffer_size_2', type=int, required=True, help="size of sec replay buffer")
    parser.add_argument('--buffer_name', type=str, required=True, help="path sec replay buffer")
    parser.add_argument('--buffer_size_new', type=int, required=True, help="size of new replay buffer")
    arg = parser.parse_args()
    main(arg)
