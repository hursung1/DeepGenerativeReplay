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

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

TrainDataLoaders = []
TestDataLoaders  = []

for i in range(5):
    # MNIST dataset
    TrainDataSet = datasets.MNIST_IncrementalDataset(source='../data/', 
                                            train=True, 
                                            transform=transform, 
                                            download=True, 
                                            classes=range(i * 2, (i+1) * 2))
                                            
    TestDataSet = datasets.MNIST_IncrementalDataset(source='../data/', 
                                           train=False, 
                                           transform=transform, 
                                           download=True, 
                                           classes=range(i * 2, (i+1) * 2))

    TrainDataLoaders.append(torch.utils.data.DataLoader(TrainDataSet, 
                                                        batch_size=batch_size, 
                                                        shuffle=True))
    TestDataLoaders.append(torch.utils.data.DataLoader(TestDataSet, 
                                                       batch_size=batch_size, 
                                                       shuffle=False))

device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
ld = 10
epochs = 200
gan_p_real_list = []
gan_p_fake_list = []
solver_acc_dict = {}

pre_gen = None
pre_solver = None

for t in range(5):
    ratio = 1 / (t+1) # current task's ratio 
    if t > 0:
        pre_gen = gen
        pre_solver = solver

        lib.model_grad_switch(pre_gen, False)
        lib.model_grad_switch(pre_solver, False)

    gen = models.Generator_Conv(input_node_size=num_noise, output_shape=(1, 28, 28), hidden_node_size=256,hidden_node_num=2)
    disc = models.Discriminator_Conv(input_shape=(1, 28, 28))
    solver = models.Solver(t+1)

    if torch.cuda.is_available():
        gen = gen.to(device)
        disc = disc.to(device)
        solver = solver.to(device)
    
    lib.init_params(gen)
    lib.init_params(disc)
    lib.init_params(solver)
    
    TrainDataLoader = TrainDataLoaders[t]

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
            
            if torch.cuda.is_available():
                x = x.to(device)
                noise = noise.to(device)

            if pre_gen is not None:
                with torch.no_grad():
                    # append generated image & label from previous scholar
                    datapart = int(num_data*ratio)
                    perm = torch.randperm(num_data)[:datapart]
                    x = x[perm]

                    x_g = pre_gen(lib.sample_noise(num_data, num_noise, device))
                    perm = torch.randperm(num_data)[:num_data - datapart]
                    x_g = x_g[perm]

                    x = torch.cat((x, x_g))
                
            ### Discriminator train
            optim_d.zero_grad()
            disc.zero_grad()
            x_g = gen(noise)

            ## Regularization term
            eps = torch.rand(1).item()
            x_hat = x.detach().clone() * eps + x_g.detach().clone() * (1 - eps)
            x_hat.requires_grad = True

            loss_xhat = disc(x_hat)
            fake = torch.ones(loss_xhat.shape[0], 1).requires_grad_(False)
            if torch.cuda.is_available():
                fake = fake.to(device)
                
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

            noise = lib.sample_noise(64, num_noise, device)
            gen_image = gen(noise)
            torchvision.utils.save_image(gen_image, 'imgs/Task_%d/%03d.png'%(t+1, epoch+1))
            
    # train solver
    for image, label in TrainDataLoader:
        celoss = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            image = image.to(device)
            label = label.to(device)
            celoss = celoss.to(device)

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