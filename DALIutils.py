#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   DALIutils.py
@Contact :   kael.sunkiller@gmail.com
@License :   (C)Copyright 2019, Leozen-Yang

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
11/21/19 2:15 PM   Yang      0.0         DALI Pipeline util for COCO Caption dataset
"""

import os
import json
import numpy as np
from time import time
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

test_data_root = '../../COCO/data'
file_root = os.path.join(test_data_root, 'train2014')
annotations_file = os.path.join(test_data_root, 'annotations', 'captions_train2014.json')

num_gpus = 1
batch_size = 16


class COCOCaptionInputIterator(object):
    def __init__(self, batch_size, img_dir, json_dir):
        '''

        Args:
            batch_size:
            img_dir:
            json_dir:

        Notes:
            This iterator supports both coco2014 and 2017 dataset, while the format of img_dir should be end with 2014
            or 2017 like "/root_path/trainorval2014". Both train and valid dir should follow this format. And the image
            name in train/valid dir should be kept as the original format as they are downloaded from the coco website.
            In dataset 2014, it should be "COCO_trainorval2014_imgid.jpg", and in dataset 2017 will be directly "imgid.jpg".

            This iterator is specifically in support of caption task using NVIDIA DALI pipeline. Other task please check
            nvidia.dali.ops.COCOReader support document.
        '''
        self.images_dir = img_dir
        self.batch_size = batch_size
        with open(json_dir, encoding='utf-8') as reader:
            self.anno_list = json.load(reader)['annotations']
        if img_dir.endswith('2014'):
            self.img_dir = os.path.join(img_dir, 'COCO_' + os.path.split(img_dir)[-1] + '_')
        else:
            self.img_dir = img_dir + '/'
        shuffle(self.anno_list)

    def __iter__(self):
        self.i = 0
        self.n = len(self.anno_list)
        return self

    def __next__(self):
        batch = []
        captions = []
        for _ in range(self.batch_size):
            img_id = self.anno_list[self.i]['image_id']
            caption = self.anno_list[self.i]['caption'].lower()
            f = open(self.img_dir + str(img_id).zfill(12) + '.jpg', 'rb')
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            captions.append(caption.encode('utf-8'))
            self.i = (self.i + 1) % self.n
        return batch, captions

    next = __next__


class COCOCaptionPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, img_dir, json_dir):
        '''

        Args:
            batch_size: batch size for output at the first dim.
            num_threads: number of cpu working threads.
            device_id: the slice number of gpu.
            img_dir: dir path where the images are stored.
            json_dir: json path for coco dataset.

        Output:
            (images, captions) pair stacked by batch_size. The output shape of images will be NHWC with type of float.
            Notes that the output type of captions will be a list of numpy which is encoded from the original string
            caption. To use it in the custom model, one needs to decode the numpy into string by .tostring() function
            or .tobytes().decode() function. .tostring will get a bytes type result while .tobytes.decode will directly
            get the string.
        '''
        super(COCOCaptionPipeline, self).__init__(batch_size, num_threads, device_id, seed=15)
        self.iterator = iter(COCOCaptionInputIterator(batch_size, img_dir, json_dir))
        self.input = ops.ExternalSource()
        self.caption = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)
        self.cast = ops.Cast(device="cpu", dtype=types.FLOAT)

    def define_graph(self):
        self.inputs = self.input()
        self.captions = self.caption()
        images = self.decode(self.inputs)
        output = self.cast(images)
        return output, self.captions

    def iter_setup(self):
        images, captions = self.iterator.next()
        self.feed_input(self.inputs, images)
        self.feed_input(self.captions, captions)


# example
if __name__ == '__main__':
    start = time()
    pipes = [COCOCaptionPipeline(batch_size, 2, device_id, file_root, annotations_file) for device_id in range(num_gpus)]
    for pipe in pipes:
        pipe.build()
    total_time = time() - start
    print("Computation graph built and dataset loaded in %f seconds." % total_time)

    pipe_out = [pipe.run() for pipe in pipes]
    images_cpu = pipe_out[0][0].at(0)
    captions_cpu = pipe_out[0][1].at(0).tobytes().decode()

    print(type(images_cpu), images_cpu.shape, captions_cpu)
