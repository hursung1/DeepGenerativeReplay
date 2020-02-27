import torch
import torchvision
import numpy as np
import lib

def train(**kwargs):
#def train(cur_task, gen, disc, solver, pre_gen, pre_solver, ratio, epochs):
    TrainDataLoaders = kwargs['TrainDataLoaders']
    TestDataLoaders = kwargs['TestDataLoaders']
    batch_size = kwargs['batch_size']
    num_noise = kwargs['num_noise']
    cur_task = kwargs['cur_task']
    gen = kwargs['gen']
    disc = kwargs['disc']
    solver = kwargs['solver']
    pre_gen = kwargs['pre_gen']
    pre_solver = kwargs['pre_solver']
    ratio = kwargs['ratio']
    epochs = kwargs['epochs']
    
    
    assert (ratio >=0 or ratio <= 1)
    bceloss = torch.nn.BCELoss()
    celoss = torch.nn.CrossEntropyLoss()

    gen_optim = torch.optim.Adam(gen.parameters(), lr=0.0002)
    disc_optim = torch.optim.Adam(disc.parameters(), lr=0.0002)
    solver_optim = torch.optim.Adam(solver.parameters(), lr=0.0001)
    train_dataloader = TrainDataLoaders[cur_task]
    
    # GAN Training
    for epoch in range(epochs):
        for image, label in train_dataloader:
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()

            if pre_gen is not None:
                # append generated image & label from previous scholar
                gen_image = pre_gen(lib.sample_noise(batch_size, num_noise))
                gen_label = pre_solver(gen_image).max(dim=1)[1]

                image = torch.cat((image, gen_image))
                label = torch.cat((label, gen_label))

            ### Discriminator Training
            disc_optim.zero_grad()
            p_real = disc(image.view(image.shape[0], -1 , 28, 28))
            p_fake = disc(gen(lib.sample_noise(image.shape[0], num_noise)))

            ones = torch.ones_like(p_real)
            zeros = torch.zeros_like(p_real)
            if torch.cuda.is_available():
                ones = ones.cuda()
                zeros = zeros.cuda()

            loss_d = bceloss(p_real, ones) + bceloss(p_fake, zeros)

            loss_d.backward()
            disc_optim.step()

            ### Generator Training
            gen_optim.zero_grad()
            p_fake = disc(gen(lib.sample_noise(batch_size, num_noise)))

            ones = torch.ones_like(p_fake)
            if torch.cuda.is_available():
                ones = ones.cuda()

            loss_g = bceloss(p_fake, ones)
            loss_g.backward()

            gen_optim.step()

        if epoch % 10 == 9:
            p_real, p_fake = lib.gan_evaluate(batch_size = batch_size,
                                              num_noise = num_noise,
                                              cur_task = cur_task, 
                                              gen = gen, 
                                              disc = disc, 
                                              TestDataLoaders = TestDataLoaders)
            gen_image = gen(lib.sample_noise(batch_size, num_noise))
            print("(Epoch {}/{}) p_real: {} | p_fake: {}\n".format(epoch, epochs, p_real, p_fake))
            lib.imshow_grid(gen_image)
    
    # train solver
    for image, label in train_dataloader:
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()

        output = solver(image)
        loss = celoss(output, label) * ratio

        if pre_solver is not None:
            noise = lib.sample_noise(batch_size, num_noise)
            g_image = pre_gen(noise)
            g_label = pre_solver(g_image).max(dim=1)[1]
            g_output = solver(g_image)
            loss += celoss(g_output, g_label) * (1 - ratio)

        loss.backward()
        solver_optim.step()