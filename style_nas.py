import os
import numpy
import random
import collections
import multiprocessing as mp

DIM = 32

class Model(object):
    def __init__(self):
        self.dim = DIM
        self.path = None
        self.arch = None
        self.arch_str = None 
        self.loss_r = None
        self.loss_p = None
        self.loss_m = None
        self.accuracy = None


    def make_dir(self):
        os.system('cp -r ./configs/photorealistic_model_nas ./configs/photorealistic_model_nas_%s' % self.arch_str)
        self.path = './configs/photorealistic_model_nas_%s' % self.arch_str

    def train(self):
        # This command is used to allocate computing resource and train the decoder. Please change the command ``srun -p 1080Ti --gres=gpu:1 --cpus-per-task 5 -n 1'' according to settings of your server cluster.
        os.system('srun -p 1080Ti --gres=gpu:1 --cpus-per-task 5 -n 1 python3 %s/train_decoder.py -d %s -me 2' % (self.path, self.arch_str))

    def evaluate(self):
        # This command is used to allocate computing resource and make photorealistic style transfer. Please change the command ``srun -p 1080Ti_dbg --gres=gpu:1 --cpus-per-task 5 -n 1'' according to settings of your server cluster.
        os.system('srun -p 1080Ti_dbg --gres=gpu:1 --cpus-per-task 5 -n 1 python3 %s/photo_transfer.py -d %s' % (self.path, self.arch_str))
        # This command is used to allocate computing resource and validate style-transferred results. Please change the command ``srun -p 1080Ti_dbg --gres=gpu:1 --cpus-per-task 5 -n 1'' according to settings of your server cluster.
        os.system('srun -p 1080Ti_dbg --gres=gpu:1 --cpus-per-task 5 -n 1 python3 %s/validation.py' % self.path)
        with open('%s/result.txt' % self.path, 'r') as f:
            acc = f.readline()
        this_arch = bin(self.arch)[2:]
        while len(this_arch) != DIM:
            this_arch = '0' + this_arch
        control_index = [int(i) for i in this_arch]
        self.loss_r = acc.split(' ')[0]
        self.loss_p = acc.split(' ')[1]
        self.loss_r = float(self.loss_r)
        self.loss_p = float(self.loss_p)
        self.loss_m = sum(control_index) / len(control_index)
        acc = 0.8 * self.loss_r + 0.1 * self.loss_p + 0.1 * self.loss_m
        return acc, self.loss_r, self.loss_p, self.loss_m

def random_architecture():
    return random.randint(0, 2**DIM - 1)

def mutate_arch(parent_arch):
    position = random.randint(0, DIM - 1)
    child_arch = parent_arch ^ (1 << position)
    return child_arch


if __name__ == '__main__':
    cycles = 200
    population_size = 50
    sample_size = 10
    population = collections.deque()
    history = []

    def train_val_model(i):
        model = Model()
        model.arch = random_architecture()
        model.arch_str = bin(model.arch)[2:]
        while len(model.arch_str) != DIM:
            model.arch_str = '0' + model.arch_str
        model.make_dir()
        model.train()
        model.accuracy, loss_r, loss_p, loss_m = model.evaluate()
        print('| acc: %.4f | loss_recon: %.4f | loss_perc: %.4f | loss_mode: %.4f |' % (model.accuracy, loss_r, loss_p, loss_m))
        return model

    p1 = mp.Pool()
    res = p1.map(train_val_model, range(population_size))
    p1.close()
    p1.join()
    for model in res:
        with open('./record.txt', 'a') as f:
            f.write('%s, %.4f, %.4f, %.4f, %.4f\n' % (model.arch_str, model.accuracy, model.loss_r, model.loss_p, model.loss_m))
        population.append(model)
        history.append(model)

    while len(history) < cycles:
        childs = []
        sample = []
        while len(sample) < sample_size:
            candidate = random.choice(list(population))
            sample.append(candidate)
        parent = min(sample, key=lambda i: i.accuracy)
        child = Model()
        child.arch = mutate_arch(parent.arch)
        child.arch_str = bin(child.arch)[2:]
        while len(child.arch_str) != DIM:
            child.arch_str = '0' + child.arch_str
        child.make_dir()
        child.train()
        child.accuracy, loss_r, loss_p, loss_m = child.evaluate()
        print('| acc: %.4f | loss_recon: %.4f | loss_perc: %.4f | loss_mode: %.4f |' % (child.accuracy, loss_r, loss_p, loss_m))
        childs.append(child)
        population.append(child)
        history.append(child)
        population.popleft()
        with open('./record.txt', 'a') as f:
            f.write('%s, %.4f, %.4f, %.4f, %.4f\n' % (child.arch_str, child.accuracy, child.loss_r, child.loss_p, child.loss_m))
