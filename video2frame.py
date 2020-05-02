# -*- coding: utf-8 -*-

"""
1.将视频转换为帧，存在JPEGImages里
2.给提取出来的每一帧写xml，放到Annotations里
3.将所有图片分为train/val/test，并将他们的名字放到Imageset/Main里
"""


import os
import subprocess
import numpy as np
import cv2
import ipdb


def assert_path(path, error_message):
    assert os.path.exists(path), error_message


def count_files(path, filename_starts_with=''):
    files = [f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))
                     and f.startswith(filename_starts_with)]
    return len(files)


# 将video转化为图片
def split_video(video_file, image_name_prefix, jpeg_image_path):
    # 按[1,13,25,37,...]取帧。这样的话，每帧之间就有12个间隔。12/fps(=30)=0.4s。因为按章论文<Social LSTM>的取值，是每0.4s取一帧。
    # observation_length = 3.2s（也就是8帧）, prediction_length = 4.8s（也就是12帧）

    command_extract = "select=1 n*not(mod(n\,13))"


    return subprocess.check_output('ffmpeg -i ' + os.path.abspath(video_file) + ' -vf "%s" -vsync 0 '%command_extract + image_name_prefix +'%d.jpg', shell=True, cwd=jpeg_image_path)
    # -vsync 0 每个帧及其时间戳从多路分配器传递到多路复用器
def log(message, level='info'):
    formatters = {
        'GREEN': '\033[92m',
        'END': '\033[0m',
    }
    print( ('{GREEN}<'+level+'>{END}\t' + message).format(**formatters))


def write_to_file(filename, content):
    f = open(filename, 'a')
    f.write(content+'\n')

'''
在ImageSets/Main/train.txt生成用于训练的文件名字。
这些文件名字对应/JPEGImages里面的图片
'''
def split_dataset(number_of_frames, split_ratio, file_name_prefix, scene):
    assert sum(split_ratio) <= 1, 'Split ratio cannot be more than 1.'



    train, val, test = np.array(split_ratio) * number_of_frames

    #DONE： 随机取图片的话，会导致tracking的顺序变乱。但是这好像只是为了训练权重。tracking可以不再用这里的文件列表的画就没关系。
    # test_images = random.sample(range(1, number_of_frames+1), int(test)) # random.sample('ABCD',2)从ABCD中随机取两个元素
    # val_images = random.sample(tuple(set(range(1, number_of_frames+1)) - set(test_images)), int(val))
    # train_images = random.sample(tuple(set(range(1, number_of_frames+1)) - set(test_images) - set(val_images)), int(train))

    # 按顺序分配train/val/test
    train_images = range(1, int(train)+1)
    val_images = tuple(set(range(1,int(train)+int(val)+1)) - set(train_images))
    test_images = tuple(set(range(1, number_of_frames+1)) - set(train_images) - set(val_images))

    for index in train_images:
        write_to_file(os.path.join(PATH_TO_DATA1,scene+'.train')
                    ,os.path.join(PATH_TO_SDD, scene,'images', file_name_prefix+str(index)+'.jpg'))

    for index in val_images:
        write_to_file(
            os.path.join(PATH_TO_DATA1, scene + '.val'),
            os.path.join(PATH_TO_SDD, scene, 'images',
                           file_name_prefix + str(index)+'.jpg'))

    for index in test_images:
        write_to_file(
            os.path.join(PATH_TO_DATA1, scene + '.test'),
            os.path.join(PATH_TO_SDD, scene, 'images',
                         file_name_prefix + str(index)+'.jpg'))


def annotate_frames(sdd_annotation_file, dest_path, scene, filename_prefix, number_of_frames, sofar_list):

    sdd_annotation = np.genfromtxt(sdd_annotation_file, delimiter=' ', dtype=np.str)

    first_image_path = os.path.join(PATH_TO_SDD, scene, 'images', filename_prefix+'1.jpg')
    assert_path(first_image_path, 'Cannot find the images. Trying to access: ' + first_image_path)
    first_image = cv2.imread(first_image_path)
    frame_height, frame_width, depth = first_image.shape

    id_list = []
    init_sofar_list_len = len(sofar_list)
    # 根据JPEGImages夹里对于每一张图片进行处理
    for frame_number in range(1, number_of_frames+1):
        # [sdd_annotation[:, 5]记录的就是frame的column，当其==str(frame_number)提取出值
        # [1,13,25,...] 注意此处是取第一帧的内容。但其实在annotation.txt里面是从第0帧开始计数的。
        annotations_in_frame = sdd_annotation[sdd_annotation[:, 5] == str(1+(frame_number-1)*12)]

        label_path = os.path.join(dest_path, filename_prefix + str(frame_number) + '.txt')
        f = open(label_path, 'w')
        for annotation_data in annotations_in_frame:

            if int(annotation_data[1]) >= 0 and int(annotation_data[2]) >= 0 \
                    and int(annotation_data[3]) <= frame_width and int(annotation_data[4]) <= frame_height:

                if int(annotation_data[6]) == 0: # lost. If 1, the annotation is outside of the view screen.
                    # 只对行人/自行车等进行研究
                    if annotation_data[9].replace('"', '') in ['Pedestrian', 'Biker', 'Skater']:
                        try:
                            idx = id_list.index(annotation_data[0])
                        except ValueError:
                            id_list.append(annotation_data[0])
                            idx = id_list.index(annotation_data[0])
                        classs = 0
                        identity = init_sofar_list_len + idx
                        x_center = (int(annotation_data[3]) + int(annotation_data[1])) / (frame_width * 2)
                        y_center = (int(annotation_data[4]) + int(annotation_data[2])) / (frame_height * 2)
                        bbox_width = (int(annotation_data[3]) - int(annotation_data[1])) / frame_width
                        bbox_height = (int(annotation_data[4]) - int(annotation_data[2]))/ frame_height

                        if not identity in sofar_list:
                            sofar_list.append(identity)

                        f.writelines(str(classs) + ' ' + str(identity) + ' ' + str(x_center) + ' ' + str(y_center) + ' '
                                 + str(bbox_width) + ' ' + str(bbox_height) +'\n')
                        print('successed writed')
                else:
                    # object lost in the view
                    print('object lost in the view')
                    continue
            else:
                print("数据有错误！！！！！！！！")
                print(annotation_data[1],annotation_data[2],annotation_data[3],annotation_data[4])
                continue

        f.close()
    return sofar_list

def split_and_annotate():
    if not os.path.exists(PATH_TO_SDD):
        os.mkdir(PATH_TO_SDD)
    if not os.path.exists(PATH_TO_DATA1):
        os.mkdir(PATH_TO_DATA1)

    assert_path(PATH_TO_SDD, ''.join(e for e in PATH_TO_SDD if e.isalnum()) + ' folder should be found in the cwd of this script.')
    # if num_training_images is not None and num_val_images is not None and num_testing_images is not None:
    #     share = calculate_share(num_training_images, num_val_images, num_testing_images)
        # 平均每个train/val/test video需要含有多少张图片,才能达到(num_training_images, num_val_images, num_testing_images)的要

    for scene in videos_to_be_processed:
        if not os.path.exists(os.path.join(PATH_TO_SDD, scene)):
            os.makedirs(os.path.join(PATH_TO_SDD, scene, 'images'))
            os.makedirs(os.path.join(PATH_TO_SDD, scene, 'labels_with_ids'))

        path = os.path.join(SOURCE_DATASET_ROOT, 'videos', scene)
        assert_path(path, path + ' not found.')

        sofar_list = []
        videos = videos_to_be_processed.get(scene) # videos: {0:(1,0,0),2:(0.1.0).3:(0,0,1)}
        if len(videos) > 0:
            for video_index in videos.keys():
                video_path = os.path.join(path, 'video' + str(video_index))
                assert_path(video_path, video_path + ' not found.')
                assert count_files(video_path) == 1, video_path+' should contain one file.'

                # Split video into frames
                # Check whether the video has already been made into frames
                jpeg_image_path = os.path.join(PATH_TO_SDD, scene,'images')
                image_name_prefix = scene + '_video' + str(video_index) + '_'
                video_file = os.path.join(video_path, 'video.mov')

                # Split Video
                log('Splitting ' + video_file)

                # 将video变成帧，并放在目标文件夹里
                # split_video(video_file, image_name_prefix, jpeg_image_path)

                log('Splitting ' + video_file + ' complete.')

                # Annotate
                log('Annotating frames from ' + video_file)
                sdd_annotation_file = os.path.join(SOURCE_DATASET_ROOT, 'annotations', scene,
                                                   'video' + str(video_index), 'annotations.txt')
                assert_path(sdd_annotation_file, 'Annotation file not found. '
                                                 'Trying to access ' + sdd_annotation_file)
                dest_path = os.path.join(PATH_TO_SDD, scene, 'labels_with_ids')

                # 此处统计了文件夹里面有多少张图片，在annotate_frames会对图片的总数进行循环
                number_of_frames = count_files(jpeg_image_path, image_name_prefix)



                sofar_list = annotate_frames(sdd_annotation_file, dest_path, scene, image_name_prefix, number_of_frames, sofar_list)
                log('Annotation Complete.')

                split_ratio = videos.get(video_index)  # 用于划分train/val/test

                split_dataset(number_of_frames, split_ratio, image_name_prefix, scene)
                    # 只是split_ratio将名字索引划分后放到文件里
                log('Successfully created train-val-test split.')
    log('Done.')


if __name__ == '__main__':

    ratio = (.8, .1, .1)

    videos_to_be_processed = \
        {

        'bookstore': {0: ratio, 1: ratio, 2: ratio, 3: ratio, 4: ratio, 5: ratio, 6:ratio},

        'coupa': {0: ratio, 1: ratio, 2: ratio, 3: ratio},

        'deathCircle': {0: ratio, 1: ratio, 2: ratio, 3: ratio, 4: ratio},

        'gates': {0: ratio, 1: ratio, 2: ratio, 3: ratio, 4: ratio, 5: ratio, 6: ratio,
                  7: ratio, 8: ratio},

        'hyang': {0: ratio, 1: ratio, 2: ratio, 3: ratio, 4: ratio, 5: ratio, 6: ratio,
                  7: ratio, 8: ratio, 9: ratio, 10: ratio, 11: ratio, 12: ratio,
                  13: ratio,14: ratio},

        'little': {0: ratio, 1: ratio, 2: ratio, 3: ratio},

        'nexus': {0: ratio, 1: ratio, 2: ratio, 3: ratio, 4: ratio, 5: ratio, 6: ratio,
                  7: ratio, 8: ratio, 9: ratio, 10: ratio, 11: ratio},
        'quad': {0: ratio, 1: ratio, 2: ratio, 3: ratio}
        }


    PATH_TO_SDD = '/home/jwei/PycharmProjects/Towards-Realtime-MOT/sdd'
    SOURCE_DATASET_ROOT = '/mrtstorage/datasets/stanford_campus_dataset/'
    PATH_TO_DATA1 = '/home/jwei/PycharmProjects/Towards-Realtime-MOT/data1'

    split_and_annotate()
