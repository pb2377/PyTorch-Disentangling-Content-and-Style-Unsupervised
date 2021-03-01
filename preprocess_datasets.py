import os
import glob
from PIL import Image, ImageDraw
import scipy.io
import xml.etree.ElementTree
import random


def Celeba():
    """
    Crop CelebA images to bounding box annotations
    :return:
    """
    dataset_dir = '../Datasets/CelebA/'
    image_dir = os.path.join(dataset_dir, 'Images-aligned', 'img_celeba')
    partition_path = os.path.join(dataset_dir, 'list_eval_partition.txt')
    f = open(partition_path, 'r')
    train_files = []
    val_files = []
    test_files = []
    file_list = []
    for line in f.readlines():
        line = line[:-1].split(' ')
        assert len(line) == 2
        file_list.append(line[0])
        if int(line[1]) == 1:
            val_files.append(line[0])
        elif int(line[1]) == 2:
            test_files.append(line[0])
        else:
            train_files.append(line[0])

    anno_path = os.path.join(dataset_dir, 'list_bbox_celeba.txt')
    f = open(anno_path, 'r')

    annos = {}
    # lines = f.read()
    # raise NotImplementedError
    for num, line in enumerate(f.readlines()):
        if num < 2:
            continue
        # line = line.encode("ascii", errors="ignore").decode()
        # print(line)
        if (' ') in line:
            line = line.split(' ')
            while '' in line:
                line.remove('')
            line[-1].replace('\n', '')
            annos[line[0]] = [int(i) for i in line[1:]]

    for n, path in enumerate(file_list):
        path = os.path.join(image_dir, path)
        outpath = os.path.join(os.path.dirname(path).split(os.sep)[-1], os.path.basename(path))
        outpath = os.path.join(dataset_dir, 'Images-cropped', outpath)
        if not os.path.exists(outpath):
            image = Image.open(path)
            x_min, y_min, w, h = annos[os.path.basename(path)]
            image = image.crop((x_min, y_min, max(image.size[0], x_min+w), max(image.size[1], y_min+h)))
            if not os.path.exists(os.path.dirname(outpath)):
                os.makedirs(os.path.dirname(outpath))
            image.save(outpath)
        if not n+1 % int(len(annos)//20):
            print("Completed {}/{} Images".format(n+1, len(file_list)))
    # write to txt files for ease
    write_to_txt(train_files, os.path.join(dataset_dir, 'train.txt'))
    write_to_txt(val_files, os.path.join(dataset_dir, 'val.txt'))
    write_to_txt(test_files, os.path.join(dataset_dir, 'test.txt'))


def Catfaces():
    """
    Crop Cat faces to their Landmark points +/- (k*h, k*w)
    :return:
    """
    dataset_dir = '../Datasets/Cats/'
    assert os.path.exists(dataset_dir)
    file_list = [i for i in glob.glob(os.path.join(dataset_dir, 'Images', 'CAT_*', '*.cat'))]
    train_files = []
    test_files = []
    random.shuffle(file_list)
    for n, path in enumerate(file_list):
        outpath = os.path.join(os.path.dirname(path).split(os.sep)[-1], os.path.basename(path)[:-4])
        outpath = os.path.join(dataset_dir, 'Images-cropped-sorted', outpath)
        if not os.path.exists(outpath):
            cat_im = Image.open(path[:-4])
            coords = cat_annos(path)
            delta_x = 0
            delta_y = 0
            x_min = 1e5
            y_min = 1e5
            for xv, yv in coords:
                y_min = min(y_min, yv)
                x_min = min(x_min, xv)
                for xu, yu in coords:
                    delta_x = max(delta_x, abs(xv-xu))
                    delta_y = max(delta_y, abs(yv - yu))
            leye = coords[0]
            reye = coords[1]
            # remove cats that are at extreme angles
            if abs(leye[-1] - reye[-1]) < 0.25*abs(leye[0] - reye[0]):
                sc = .05
                # y_max = min(cat_im.size[1], y_min + delta_y * (1 + 6*sc))
                x_max = min(cat_im.size[0], x_min + delta_x * (1 + sc))
                x_min = max(0, x_min - delta_x * sc)
                delta_x = (x_max - x_min) * (1 + sc)
                y_max = min(cat_im.size[1], y_min + delta_x)
                # sc = 0
                y_min = y_min
                cat_im = cat_im.crop((x_min, y_min, x_max, y_max))
                # cat_im = cat_im.resize((256, 256))
                if not os.path.exists(os.path.dirname(outpath)):
                    os.makedirs(os.path.dirname(outpath))
                cat_im.save(outpath)
        if n < int(0.75 * len(file_list)):
            train_files.append(outpath)
        else:
            test_files.append(outpath)
    print('Cropped {} Cat Faces'.format(len(file_list)))
    # write_to_txt(train_files, os.path.join(dataset_dir, 'train.txt'))
    # write_to_txt(test_files, os.path.join(dataset_dir, 'test.txt'))


def write_to_txt(list_ids, outpath):
    f = open(outpath, "w")
    for item in list_ids:
        f.write(item + ' \n')
    f.close()


def cat_annos(file):
    f = open(file, 'r')
    data_list = f.read().split(' ')[1:-1]
    coord_pairs = []
    for i in range(0, len(data_list)-1, 2):
        # print(i)
        coord_pairs.append((int(data_list[i]), int(data_list[i+1])))
    return coord_pairs


def Stanforddogs():
    """
    Crop Stanford dog images to bounding box annotations
    :return:
    """
    dataset_dir = '../Datasets/StanfordDogs/'
    assert os.path.exists(dataset_dir)
    file_list = scipy.io.loadmat(os.path.join(dataset_dir, 'lists', 'file_list.mat'))
    annos = file_list['annotation_list']
    for i, path in enumerate(annos):
        path = path[0][0]
        image_path = os.path.join(dataset_dir, 'Images', path+'.jpg')
        bbox_path = os.path.join(dataset_dir, 'Annotation', path)
        assert os.path.exists(image_path)
        assert os.path.exists(bbox_path)

        bbox = get_boxes(bbox_path)[0]
        aspect_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
        out_path = os.path.join(dataset_dir, 'Images-cropped', path + '.jpg')
        if 0.9 < aspect_ratio < 1.1 and not os.path.exists(out_path):
            image = Image.open(image_path).convert('RGB')
            image = image.crop(bbox)
            if not os.path.exists(os.path.dirname(out_path)):
                os.makedirs(os.path.dirname(out_path))
            image.save(out_path)

        if not i+1 % int(len(annos)//20):
            print("Completed {}/{} Images".format(i+1, len(annos)))

    # count cropped outputs
    cropped = glob.glob(os.path.join(dataset_dir, 'Images-cropped', '*', '*'))
    print("{} Cropped Images from {} Original Annotated Images".format(len(cropped), len(annos)))
    pass


def get_boxes(path):
    """
    Get bounding boxes for Standford Dogs
    Source: https://github.com/zrsmithson/Stanford-dogs
    """
    e = xml.etree.ElementTree.parse(path).getroot()
    boxes = []
    for objs in e.iter('object'):
        boxes.append([int(objs.find('bndbox').find('xmin').text),
                      int(objs.find('bndbox').find('ymin').text),
                      int(objs.find('bndbox').find('xmax').text),
                      int(objs.find('bndbox').find('ymax').text)])
    return boxes


if __name__ == '__main__':
    # Stanforddogs()
    Catfaces()
    Celeba()
