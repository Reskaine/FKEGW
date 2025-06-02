# %%
import time
import numpy as np
from dataset_formal import *
from network import *
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from IPython.display import clear_output
from numpy import random
from module.utils import cache_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_devices = list(np.arange(torch.cuda.device_count()))
work_space = 'D:/PycharmProject/FKEGW/'

multi_gpu = len(gpu_devices) > 1

output_folder = 'D:/PycharmProject/FKEGW//Outputs'
checkpoints_folder = 'D:/PycharmProject/FKEGW//Checkpoints'
dataset_root = os.path.join(work_space, 'Data')

# %%

train_set = 'WD'
mean_, std_ = dataset_mean_std(train_set)
ti = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_, std_)])
M = 10

info_file = 'Info__' + train_set + '.txt'
list_file = 'List__' + train_set + '.txt'
label1_file = 'Label1__' + train_set + '.txt'
label2_file = 'Label2__' + train_set + '.txt'
# knowledge_file = 'WD_knowledge.txt'

GroupSizes, GroupFileLists = dataset_parser(dataset_root, info_file, list_file)
TrainGroupFileLists = []
ValueGroupFileLists = []
TestGroupFileLists = []
labels1_value_bag = []
labels1_test_bag = []
labels2_value_bag = []
labels2_test_bag = []
lst = []
selected_indexes = []
with open(os.path.join(dataset_root, label1_file), 'r') as f:
    label1 = f.read().splitlines()
with open(os.path.join(dataset_root, label2_file), 'r') as f:
    label2 = f.read().splitlines()
for index in range(0, 800):
    lst.append(int(label1[index]))
    TrainGroupFileLists.append(GroupFileLists[index])
for category in set(lst):
    indexes = [i for i, x in enumerate(lst) if x == category]
    selected_indexes.extend(indexes)
for index in range(800, 900):
    ValueGroupFileLists.append(GroupFileLists[index])
    labels1_value_bag.append(int(label1[index]))
    labels2_value_bag.append(int(label2[index]))
for index in range(900, 1009):
    TestGroupFileLists.append(GroupFileLists[index])
    labels1_test_bag.append(int(label1[index]))
    labels2_test_bag.append(int(label2[index]))
print(len(selected_indexes))
if np.mod(len(TestGroupFileLists), M) != 0:
    gs = np.mod(len(TestGroupFileLists), M)
    for add_index in range(M - gs):
        TestGroupFileLists.append(TestGroupFileLists[add_index])
        labels1_value_bag.append(int(label1[add_index]))
        labels1_test_bag.append(int(label1[add_index]))
        labels2_value_bag.append(int(label2[add_index]))
        labels2_test_bag.append(int(label2[add_index]))
batch_group_labels1_value = torch.tensor(labels1_value_bag)
batch_group_labels1_test = torch.tensor(labels1_test_bag)
batch_group_labels2_value = torch.tensor(labels2_value_bag)
batch_group_labels2_test = torch.tensor(labels2_test_bag)
group_file_list_value = ValueGroupFileLists
group_file_list_test = TestGroupFileLists

net = FKEGW(block=BasicBlock, num_block=[3, 4, 6, 3], in_channels=3, out_channels=[64, 128, 256, 512], num_classes=5,
            img_size=224, fpn_cell_repeats=3).cuda()

epoch = 100
iterations = 80
show_every = iterations
init_lr = 8e-4
min_lr = 5e-4
optimizer = optim.Adam(net.parameters(), lr=init_lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(iterations), eta_min=min_lr)
writer = SummaryWriter("D:/PycharmProject/FKEGW/logs")
should_break = False
# %%
net.train()
max_value = 0.0
for ep in range(1, epoch + 1):
    random.shuffle(selected_indexes)
    print(selected_indexes)
    loss1 = 0.0
    for it in range(1, iterations + 1):
        time_start = time.time()
        grp_images, grp_labels1, grp_labels2, grp_names = batch_group_loader(dataset_root, TrainGroupFileLists,
                                                                             label1_file, label2_file, M, mean_, std_,
                                                                             it, selected_indexes)
        grp_images, grp_labels1, grp_labels2 = grp_images.cuda(), grp_labels1.cuda(), grp_labels2.cuda()
        optimizer.zero_grad()
        label, loss_, loss_classification, loss_c, loss_d, loss_e, loss_semantic = net(grp_images, grp_labels1,
                                                                                       grp_labels2, M,
                                                                                       (ep - 1) * iterations + it)
        label1 = torch.argmax(label, dim=1)
        label1 = label1.tolist()
        loss = loss_
        loss1 += loss_.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_test = 0.0
        writer.add_scalar('Train Loss', loss.item(), (ep - 1) * iterations + it)
        writer.add_scalar('Train Loss/classification', loss_classification.item(), (ep - 1) * iterations + it)
        writer.add_scalar('Train Loss/c_rule', loss_c.item(), (ep - 1) * iterations + it)
        writer.add_scalar('Train Loss/d_rule', loss_d.item(), (ep - 1) * iterations + it)
        writer.add_scalar('Train Loss/e_rule', loss_e.item(), (ep - 1) * iterations + it)
        writer.add_scalar('Train Loss/semantic', loss_semantic.item(), (ep - 1) * iterations + it)
        for i in range(0, M):
            writer.add_scalar('Train Predict/' + str(i), label1[i], (ep - 1) * iterations + it)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], (ep - 1) * iterations + it)
        if np.mod((ep - 1) * iterations + it, show_every) == 0:
            net.eval()
            s = 0
            for index in range(0, len(group_file_list_value), M):
                images_bag = []
                local_path = group_file_list_value[index:index + M]
                for j in range(0, M):
                    images_bag.append(
                        ti(Image.open(os.path.join(dataset_root, local_path[j]) + '.bmp')).unsqueeze(0))
                with torch.no_grad():
                    grp_images_ = torch.cat(images_bag, dim=0).cuda()
                    grp_labels1_ = batch_group_labels1_value[index:index + M].view(M).cuda()
                    grp_labels2_ = batch_group_labels2_value[index:index + M].view(M).cuda()
                    label, _, _, _, _, _, _ = net(grp_images_, grp_labels1_, grp_labels2_, M,
                                                  (ep - 1) * iterations + it)
                    max_index = torch.argmax(label, dim=1)
                    for x, y in zip(max_index, grp_labels1_):
                        if x == y:
                            a = 1
                        else:
                            a = 0
                        s += a
            value_accuracy = s / (len(batch_group_labels1_value))
            print('epoch {} value accuracy : {} %'.format(ep, value_accuracy * 100))
            writer.add_scalar("Value Accuracy", value_accuracy, ((ep - 1) * iterations + it))
            s = 0
            for index in range(0, len(group_file_list_test), M):
                images_bag = []
                local_path = group_file_list_test[index:index + M]
                for j in range(0, M):
                    images_bag.append(
                        ti(Image.open(os.path.join(dataset_root, local_path[j]) + '.bmp')).unsqueeze(0))
                with torch.no_grad():
                    grp_images_ = torch.cat(images_bag, dim=0).cuda()
                    grp_labels1_ = batch_group_labels1_test[index:index + M].view(M).cuda()
                    grp_labels2_ = batch_group_labels2_test[index:index + M].view(M).cuda()
                    label, loss_t, loss_classification1, loss_c1, loss_d1, loss_e1, loss_semantic = net(grp_images_,
                                                                                                        grp_labels1_,
                                                                                                        grp_labels2_, M,
                                                                                                        (ep - 1) * iterations + it)
                    loss_test += loss_classification1
                    max_index = torch.argmax(label, dim=1)
                    for x, y in zip(max_index, grp_labels1_):
                        if x == y:
                            a = 1
                        else:
                            a = 0
                        s += a
            test_accuracy = s / (len(batch_group_labels1_test))
            print('epoch {} test accuracy : {} %'.format(ep, test_accuracy * 100))
            if test_accuracy > max_value:
                max_value = test_accuracy
                cache_model(net, os.path.join(checkpoints_folder, 'trained', 'FKEGW' + str((ep - 1) *
                            iterations + it) + '.pth'), multi_gpu)
            loss_test = loss_test * M / (len(batch_group_labels1_test))
            writer.add_scalar("Test Loss", loss_test.item(), ((ep - 1) * iterations + it))
            writer.add_scalar("Test Accuracy", test_accuracy, ((ep - 1) * iterations + it))
            if loss_test < 4:
                cache_model(net, os.path.join(checkpoints_folder, 'warehouse', 'FKEGW' + str((ep - 1) *
                            iterations + it) + '.pth'), multi_gpu)
            net.train()
            writer.add_scalar("Train Loss/epoch", loss1 / show_every, ((ep - 1) * iterations + it))
            if loss1 / show_every < 5e-2:
                should_break = True
                break
        time_end = time.time()
        if it <= 10:
            print(
                'epoch: {} running time per iteration {}: {} seconds'.format(ep, it, np.around((time_end - time_start),
                                                                                               2)))
        if it == 11:
            clear_output()
    if should_break:
        print('Process end')
        break

writer.close()

# %%


# %%
