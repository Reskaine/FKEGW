import os
from PIL import Image
import torch
from torchvision import transforms


def dataset_mean_std(dataset):
    assert dataset in ['WD', 'WD1', 'WD2']
    if dataset == 'WD':
        mean_, std_ = [0.350867, 0.350867, 0.350867], [0.150654, 0.150654, 0.150654]
    if dataset == 'WD1':
        mean_, std_ = [0.322610, 0.322610, 0.322610], [0.135263, 0.135263, 0.135263]
    if dataset == 'WD2':
        mean_, std_ = [0.345517, 0.345517, 0.345517], [0.130122, 0.130122, 0.130122]
    return mean_, std_


def dataset_parser(dataset_root, info_file, list_file):
    with open(os.path.join(dataset_root, info_file), 'r') as f:
        contents = f.read().splitlines()
    group_sizes = []
    for c in contents:
        g_size = int(c)
        group_sizes.append(g_size)
    with open(os.path.join(dataset_root, list_file), 'r') as f:
        local_paths = f.read().splitlines()
    start_index = 0
    end_index = group_sizes[0]
    group_file_lists = []
    for index in range(start_index, end_index):
        group_file_lists.append(local_paths[index])

    return group_sizes, group_file_lists


def batch_group_loader(dataset_root, group_file_lists, label_file1, label_file2, b, dataset_mean, dataset_std, k,
                       image_id):
    ti = transforms.Compose([transforms.ToTensor(), transforms.Normalize(dataset_mean, dataset_std)])

    batch_group_images_bag = []
    batch_group_names_bag = []
    group_file_list = group_file_lists
    image_ids = []
    for i in range(0, b):
        image_id_ = image_id[(k-1)*b+i]
        image_ids.append(image_id_)
    images_bag = []
    names_bag = []
    labels1_bag = []
    labels2_bag = []
    with open(os.path.join(dataset_root, label_file1), 'r') as f:
        label1 = f.read().splitlines()
    with open(os.path.join(dataset_root, label_file2), 'r') as f:
        label2 = f.read().splitlines()
    for num, i in enumerate(image_ids):
        local_path = group_file_list[i]
        names_bag.append(local_path)
        images_bag.append(ti(Image.open(os.path.join(dataset_root, local_path + '.bmp'))).unsqueeze(0))
        labels1_bag.append(int(label1[i]))
        labels2_bag.append(int(label2[i]))

    batch_group_images_bag.append(torch.cat(images_bag, dim=0))
    batch_group_names_bag.append(names_bag)
    batch_group_labels1 = torch.tensor(labels1_bag)
    batch_group_labels2 = torch.tensor(labels2_bag)
    batch_group_images = torch.cat(batch_group_images_bag, dim=0)
    return batch_group_images, batch_group_labels1, batch_group_labels2, batch_group_names_bag

