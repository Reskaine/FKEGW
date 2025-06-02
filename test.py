# %%
import torch
import time
from dataset_formal import *
from network import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_devices = list(np.arange(torch.cuda.device_count()))
work_space = 'D:/PycharmProject/FKEGW/'

multi_gpu = len(gpu_devices) > 1

output_folder = 'D:/PycharmProject/FKEGW/Outputs'
checkpoints_folder = 'D:/PycharmProject/FKEGW/Checkpoints'
dataset_root = os.path.join(work_space, 'Data')

test_set = 'WD'
mean_, std_ = dataset_mean_std(test_set)
ti = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_, std_)])
M = 1

info_file = 'Info__' + test_set + '.txt'
list_file = 'List__' + test_set + '.txt'

GroupSizes, GroupFileLists = dataset_parser(dataset_root, info_file, list_file)
TrainGroupFileLists = []
ValueGroupFileLists = []
TestGroupFileLists = []
for index in range(0, 800):
    TrainGroupFileLists.append(GroupFileLists[index])
for index in range(800, 900):
    ValueGroupFileLists.append(GroupFileLists[index])
for index in range(900, 1009):
    TestGroupFileLists.append(GroupFileLists[index])
if np.mod(len(TestGroupFileLists), M) != 0:
    gs = np.mod(len(TestGroupFileLists), M)
    for add_index in range(M - gs):
        ValueGroupFileLists.append(TestGroupFileLists[add_index])
        TestGroupFileLists.append(TestGroupFileLists[add_index])

net = FKEGW(block=BasicBlock, num_block=[3, 4, 6, 3], in_channels=3, out_channels=[64, 128, 256, 512], num_classes=5,
            img_size=224, fpn_cell_repeats=3).cuda()
resume_net_params = os.path.join(checkpoints_folder, 'trained', 'FKEGW1360.pth')

net.load_state_dict(torch.load(resume_net_params))
# %%
net.eval()
group_file_list = TrainGroupFileLists + ValueGroupFileLists + TestGroupFileLists
output_path1 = os.path.join(output_folder, test_set + '_predict.txt')
output_path2 = os.path.join(output_folder, test_set + '_tsne_input.txt')
total_time = []
for index in range(0, len(group_file_list), M):
    time_start = time.time()
    images_bag = []
    local_path = group_file_list[index:index + M]
    for j in range(0, M):
        images_bag.append(
            ti(Image.open(os.path.join(dataset_root, local_path[j]) + '.bmp')).unsqueeze(0))
    with torch.no_grad():
        grp_images_ = torch.cat(images_bag, dim=0).cuda()
        grp_labels1 = torch.zeros(M).to(torch.int64).cuda()
        grp_labels2 = torch.zeros(M).to(torch.int64).cuda()
        feature, label, _, _, _, _, _, _ = net(grp_images_, grp_labels1, grp_labels2, M, 50000)
        max_index = torch.argmax(label, dim=1)
        max_index = max_index.tolist()
    time_end = time.time()
    total_time.append(np.around((time_end - time_start), 4))
    feature = feature.view(-1).tolist()
    features = ' '.join(map(str, feature))
    print('running time per frame {}: {} seconds'.format(index, np.around((time_end - time_start), 4)))
    if index == 0:
        with open(output_path1, 'w', encoding='utf-8') as file1:
            for i in range(M):
                print(max_index[i], file=file1)
    else:
        with open(output_path1, 'a', encoding='utf-8') as file2:
            for i in range(M):
                print(max_index[i], file=file2)
    if index == 0:
        with open(output_path2, 'w', encoding='utf-8') as file3:
            for i in range(M):
                print(features, file=file3)
    else:
        with open(output_path2, 'a', encoding='utf-8') as file4:
            for i in range(M):
                print(features, file=file4)

times = total_time[1:]
totaltime = sum(times) / len(times)
print('total running time per frame : {} fps'.format(1 / totaltime))
print('speed per frame : {} ms'.format(totaltime * 1000))
# %%

#import os

#os.system('rm -rf ./Outputs/Cosal2015/*.png')
#os.system('rm -rf ./Outputs/iCoseg/*.png')
#os.system('rm -rf ./Outputs/MSRC/*.png')
