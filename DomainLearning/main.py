import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import pyfiles.datasets as datasets
import pyfiles.models as models
import pyfiles.lib as lib
import pyfiles.train as train

batch_size = num_noise = 64
TrainDataSets = []
TestDataSets = []
DataShape = []

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Normalize([0.5], [0.5])])

MNISTTrainDataset = torchvision.datasets.MNIST(
                        root='../data',
                        train=True,
                        transform=transform,
                        download=True
                    )
MNISTTestDataset = torchvision.datasets.MNIST(
                        root='../data',
                        train=False,
                        transform=transform,
                        download=True
                    )

TrainDataSets.append(MNISTTrainDataset)
TestDataSets.append(MNISTTestDataset)
DataShape.append((1, 28, 28))

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

SVHNTrainDataset = torchvision.datasets.SVHN(
                        root='../data',
                        split='train',
                        transform=transform,
                        download=True
                    )
SVHNTestDataset = torchvision.datasets.SVHN(
                        root='../data',
                        split='test',
                        transform=transform,
                        download=True
                    )

TrainDataSets.append(SVHNTrainDataset)
TestDataSets.append(SVHNTestDataset)
DataShape.append((3, 32, 32))


device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
ld = 10
epochs = 200
gan_p_real_list = []
gan_p_fake_list = []
solver_acc_dict = {}

pre_gen = None
pre_solver = None

for t in range(1):
    ratio = 1 / (t+1) # current task's ratio 

    TrainDataSet = TrainDataSets[1]
    TrainDataLoader = torch.utils.data.DataLoader(
        TrainDataSet,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    data_shape = DataShape[1]

    if t > 0:
        pre_gen = gen
        pre_solver = solver

        lib.model_grad_switch(pre_gen, False)
        lib.model_grad_switch(pre_solver, False)

    gen = models.Generator_Conv(input_node_size=num_noise, output_shape=data_shape).to(device)
    disc = models.Discriminator_Conv(input_shape=data_shape).to(device)
    solver = models.Solver(t+1).to(device)

    lib.init_params(gen)
    lib.init_params(disc)
    lib.init_params(solver)
    
    optim_g = torch.optim.Adam(gen.parameters(), lr=0.001, betas=(0, 0.9))
    optim_d = torch.optim.Adam(disc.parameters(), lr=0.001, betas=(0, 0.9))
    optim_s = torch.optim.Adam(solver.parameters(), lr=0.001)

    # Generator Training
    for epoch in range(epochs):
        gen.train()
        disc.train()

        for i, (x, _) in enumerate(TrainDataLoader):
            num_data = x.shape[0]
            noise = lib.sample_noise(num_data, num_noise, device)
            x = x.to(device)

            if pre_gen is not None:
                with torch.no_grad():
                    # append generated image & label from previous scholar
                    x_g = pre_gen(lib.sample_noise(batch_size, num_noise, device))
                    x = torch.cat((x, x_g))
                    perm = torch.randperm(x.shape[0])[:num_data]
                    x = x[perm]
                
            ### Discriminator train
            optim_d.zero_grad()
            disc.zero_grad()
            x_g = gen(noise)

            ## Regularization term
            
            eps = torch.rand(1).item()
            x_hat = x.detach().clone() * eps + x_g.detach().clone() * (1 - eps)
            x_hat.requires_grad = True

            loss_xhat = disc(x_hat)
            fake = torch.ones(loss_xhat.shape[0], 1).requires_grad_(False).to(device)
                
            gradients = torch.autograd.grad(outputs = loss_xhat,
                                            inputs = x_hat,
                                            grad_outputs=fake,
                                            create_graph = True,
                                            retain_graph = True,
                                            only_inputs = True)[0]
            gradients = gradients.view(gradients.shape[0], -1)
            gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * ld
            
            p_real = disc(x)
            p_fake = disc(x_g.detach())

            loss_d = torch.mean(p_fake) - torch.mean(p_real) + gp
            loss_d.backward()
            optim_d.step()

            ### Generator Training
            if i % 5 == 4:
                gen.zero_grad()
                optim_g.zero_grad()
                p_fake = disc(x_g)

                loss_g = -torch.mean(p_fake)
                loss_g.backward()
                optim_g.step()

        print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (epoch+1, epochs, loss_d.item(), loss_g.item()))
        if epoch % 10 == 9:
            dir_name = "imgs/Task_%d" % (t+1)
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)

            gen_image = gen(lib.sample_noise(64, num_noise, device))
            torchvision.utils.save_image(gen_image, 'imgs/Task_%d/%03d.png'%(t+1, epoch+1))
            #lib.imsave(gen_image, epoch, path=dir_name)
    
    # train solver
    for image, label in TrainDataLoader:
        celoss = torch.nn.CrossEntropyLoss().to(device)
        image = image.to(device)
        label = label.to(device)

        solver.zero_grad()
        optim_s.zero_grad()

        output = solver(image)
        loss = celoss(output, label) * ratio
        loss.backward()
        optim_s.step()

        if pre_solver is not None:
            solver.zero_grad()
            optim_s.zero_grad()

            noise = lib.sample_noise(batch_size, num_noise, device)
            g_image = pre_gen(noise)
            g_label = pre_solver(g_image).max(dim=1)[1]
            g_output = solver(g_image)
            loss = celoss(g_output, g_label) * (1 - ratio)

            loss.backward()
            optim_s.step()
    
    ### Evaluate solver
    solver_acc_dict[t+1] = lib.solver_evaluate(t, gen, solver, ratio, device, TestDataLoaders)
    
x, y = list(solver_acc_dict.keys()), list(solver_acc_dict.values())
plt.xlabel('task #')
plt.ylabel('accuracy')
plt.plot(x, y)
plt.savefig('result.png', dpi=300)