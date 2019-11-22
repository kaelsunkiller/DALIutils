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
import torch
import ctypes
import logging
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
batch_size = 64

MEAN = [i * 255 for i in [0.485, 0.456, 0.406]]
STD = [i * 255 for i in [0.229, 0.224, 0.225]]


class COCOCaptionInputIterator(object):
    def __init__(self, batch_size, device_id, num_gpus, img_dir, json_dir, shuffle=True):
        """

        Args:
            batch_size:
            device_id:
            num_gpus:
            img_dir:
            json_dir:

        Notes:
            This iterator supports both coco2014 and 2017 dataset, while the format of img_dir should be end with 2014
            or 2017 like "/root_path/trainorval2014". Both train and valid dir should follow this format. And the image
            name in train/valid dir should be kept as the original format as they are downloaded from the coco website.
            In dataset 2014, it should be "COCO_trainorval2014_imgid.jpg", and in dataset 2017 will be directly "imgid.jpg".

            This iterator is specifically in support of caption task using NVIDIA DALI pipeline. Other task please check
            nvidia.dali.ops.COCOReader support document.
        """
        self.images_dir = img_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        with open(json_dir, encoding='utf-8') as reader:
            self.anno_list = json.load(reader)['annotations']
        if img_dir.endswith('2014'):
            self.img_dir = os.path.join(img_dir, 'COCO_' + os.path.split(img_dir)[-1] + '_')
        else:
            self.img_dir = img_dir + '/'
        self.data_set_len = len(self.anno_list)
        self.anno_list = self.anno_list[self.data_set_len * device_id // num_gpus:
                                        self.data_set_len * (device_id + 1) // num_gpus]
        self.n = len(self.anno_list)

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            shuffle(self.anno_list)
        return self

    def __next__(self):
        batch = []
        captions = []
        img_ids = []
        if self.i >= self.n:
            raise StopIteration
        for _ in range(self.batch_size):
            img_id = self.anno_list[self.i]['image_id']
            id = self.anno_list[self.i]['id']
            caption = self.anno_list[self.i]['caption'].lower()
            f = open(self.img_dir + str(img_id).zfill(12) + '.jpg', 'rb')
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            captions.append(caption.encode('utf-8'))
            img_ids.append(str(id).encode('utf-8'))
            self.i = (self.i + 1) % self.n
        return batch, captions, img_ids

    @property
    def epoch_size(self, ):
        return self.n // self.batch_size

    @property
    def size(self, ):
        return self.n

    @property
    def total_size(self, ):
        return self.data_set_len

    next = __next__


class COCOCaptionPipeline(Pipeline):
    def __init__(self, batch_size, img_dir, json_dir, num_threads=2, device_id=0, num_gpus=1, resize=None,
                 augment=False, shuffle=True):
        """

        Args:
            batch_size: batch size for output at the first dim.
            num_threads: int, number of cpu working threads.
            device_id: int, the slice number of gpu.
            num_gpus: int, number of multiple gpu.
            img_dir: str, dir path where the images are stored.
            json_dir: str, json path for coco dataset.
            resize(optional): default int, if other format please modify function params in ops.Resize.

        Output:
            (images, captions) pair stacked by batch_size. The output shape of images will be NCHW with type of float.
            Note that the output type of captions will be a list of numpy which is encoded from the original string
            caption. To use it in the custom model, one needs to decode the numpy into string by .tostring() function
            or .tobytes().decode() function. .tostring will get a bytes type result while .tobytes.decode will directly
            get the string.

        Notes:
            param 'device' in ops functions instruct which device will process the data. optional in 'mixed'/'cpu'/'gpu',
            for detail please see DALI documentation online.

        """
        super(COCOCaptionPipeline, self).__init__(batch_size, num_threads, device_id, seed=15)
        self.coco_itr = COCOCaptionInputIterator(batch_size, device_id, num_gpus, img_dir, json_dir, shuffle=shuffle)
        self.iterator = iter(self.coco_itr)
        self.input = ops.ExternalSource()
        self.caption = ops.ExternalSource()
        self.img_id = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.augment = augment

        if resize is not None:
            if isinstance(resize, tuple):
                resx, resy = resize
            elif isinstance(resize, int):
                resx = resy = resize
            else:
                resx = resy = 0.
            self.res = ops.Resize(device="gpu", resize_x=resx, resize_y=resy)
        else:
            self.res = None
        if augment:
            self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                                output_dtype=types.FLOAT,
                                                output_layout=types.NHWC,
                                                image_type=types.RGB,
                                                mean=MEAN,
                                                std=STD)
            self.cf = ops.CoinFlip()
            self.rotate = ops.Rotate(device="gpu")
            self.rng = ops.Uniform(range=(-10.0, 10.0))

    def define_graph(self):
        self.inputs = self.input()
        self.captions = self.caption()
        self.img_ids = self.img_id()
        images = self.decode(self.inputs)
        if self.res:
            images = self.res(images)
        if self.augment:
            p = self.cf()
            images = self.cmnp(images, mirror=p)
            angle = self.rng()
            images = self.rotate(images, angle=angle)
        output = images
        return output, self.captions, self.img_ids

    def iter_setup(self):
        try:
            images, captions, img_ids = self.iterator.next()
            self.feed_input(self.inputs, images)
            self.feed_input(self.captions, captions)
            self.feed_input(self.img_ids, img_ids)
        except StopIteration:
            self.iterator = iter(self.coco_itr)
            raise StopIteration

    def epoch_size(self, name=None):
        return self.coco_itr.epoch_size

    def size(self):
        return self.coco_itr.size


to_torch_type = {
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.int8): torch.int8,
    np.dtype(np.int16): torch.int16,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64
}


def feed_ndarray(dali_tensor, arr):
    """
    Copy contents of DALI tensor to pyTorch's Tensor.
    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    `arr` : torch.Tensor
            Destination of the copy
    """
    assert dali_tensor.shape() == list(arr.size()), \
        ("Shapes do not match: DALI tensor has size {0}"
         ", but PyTorch Tensor has size {1}".format(dali_tensor.shape(), list(arr.size())))
    # turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    dali_tensor.copy_to_external(c_type_pointer)
    return arr


class DALICOCOIterator(object):
    """
    COCO DALI iterator for pyTorch.
    Parameters
    ----------
    pipelines : list of nvidia.dali.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.

    Outputs
    ----------
    outputs format: list[num_gpus * dict{"images": Tensor(N, H, W, C) "captions": list[str1, str2, ..., str_batchsize]}]
    example:
        for i, data in enumerate(pipeline_iterator):
            images = data[device_id]["images"].permute(0, 3, 1, 2)  # for pytorch usage inputs format should be NCHW
            captions = data[device_id]["captions"]  # batch size length list of string
    """

    def __init__(self, pipelines, size, auto_reset=False):
        if not isinstance(pipelines, list):
            pipelines = [pipelines]

        self._num_gpus = len(pipelines)
        assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        self.batch_size = pipelines[0].batch_size
        self._size = size
        self._pipes = pipelines
        self._auto_reset = auto_reset

        # Build all pipelines
        for p in self._pipes:
            p.build()

        # Use double-buffering of data batches
        self._data_batches = [[None, None] for i in range(self._num_gpus)]
        self._counter = 0
        self._current_data_batch = 0
        self.output_map = ["images", "captions", "image_ids"]

        # We need data about the batches (like shape information),
        # so we need to run a single batch as part of setup to get that info
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR) as check:
                p.schedule_run()
        self._first_batch = None
        self._first_batch = self.next()

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        if self._counter > self._size:
            if self._auto_reset:
                self.reset()
            raise StopIteration

        # Gather outputs
        outputs = []
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR) as check:
                outputs.append(p.share_outputs())
        for i in range(self._num_gpus):
            dev_id = self._pipes[i].device_id
            out_images = []
            out_captions = []
            out_img_ids = []
            # segregate outputs into image/captions entries
            for j, out in enumerate(outputs[i]):
                if self.output_map[j] == "images":
                    out_images = out
                elif self.output_map[j] == "captions":
                    out_captions = out
                elif self.output_map[j] == "image_ids":
                    out_img_ids = out

            # Change DALI TensorLists into Tensors
            images = out_images.as_tensor()
            images_shape = images.shape()
            captions = [out_captions.at(i).tobytes().decode() for i in range(images_shape[0])]
            img_ids = [out_img_ids.at(i).tobytes().decode() for i in range(images_shape[0])]

            torch.cuda.synchronize()

            # We always need to alocate new memory as bboxes and labels varies in shape
            images_torch_type = to_torch_type[np.dtype(images.dtype())]

            torch_gpu_device = torch.device('cuda', dev_id)

            pyt_images = torch.zeros(images_shape, dtype=images_torch_type, device=torch_gpu_device)

            self._data_batches[i][self._current_data_batch] = {"images": pyt_images, "captions": captions,
                                                               "image_ids": img_ids}

            # Copy data from DALI Tensors to torch tensors
            # for j, i_arr in enumerate(images):
            feed_ndarray(images, pyt_images)

        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR) as check:
                p.release_outputs()
                p.schedule_run()

        copy_db_index = self._current_data_batch
        # Change index for double buffering
        self._current_data_batch = (self._current_data_batch + 1) % 2
        self._counter += self._num_gpus * self.batch_size
        return [db[copy_db_index] for db in self._data_batches]

    def next(self):
        """
        Returns the next batch of data.
        """
        return self.__next__();

    def __iter__(self):
        return self

    def reset(self):
        """
        Resets the iterator after the full epoch.
        DALI iterators do not support resetting before the end of the epoch
        and will ignore such request.
        """
        if self._counter > self._size:
            self._counter = self._counter % self._size
        else:
            logging.warning("DALI iterator does not support resetting while epoch is not finished. Ignoring...")


# example

def numpy_test():
    start = time()
    pipes = [COCOCaptionPipeline(batch_size, file_root, annotations_file, device_id=device_id, resize=256, augment=True,
                                 shuffle=False)
             for device_id in range(num_gpus)]
    for pipe in pipes:
        pipe.build()
    total_time = time() - start
    print("Computation graph built and dataset loaded in %f seconds." % total_time)

    pipe = pipes[0]
    img_ids = []
    start = time()
    for e in range(1):
        print("epoch: ", e)
        for itr in range(pipe.epoch_size()):
            c_time = time()
            pipe_out = pipe.run()
            images_cpu = pipe_out[0].as_cpu().as_array()
            captions_cpu = [pipe_out[1].at(i).tobytes().decode() for i in range(batch_size)]
            img_ids_cpu = [pipe_out[2].at(i).tobytes().decode() for i in range(batch_size)]
            img_ids.extend(img_ids_cpu)
            time_used = time() - c_time
            print("itr: {}, img ids: {}, data type: {}, data shape: {}, captions: {}, time: {}".format(itr, img_ids_cpu[0],
                                                                                             type(images_cpu),
                                                                                             images_cpu.shape,
                                                                                             len(captions_cpu), time_used))
    total_time = time() - start
    print("time: {}".format(total_time), len(img_ids), len(set(img_ids)))


def tensor_test():
    start = time()
    pipe = COCOCaptionPipeline(batch_size, file_root, annotations_file, resize=256, augment=True, shuffle=False)
    pii = DALICOCOIterator(pipe, size=pipe.size(), auto_reset=True)
    total_time = time() - start
    print("Computation graph built and dataset loaded in %f seconds." % total_time)

    start = time()
    for e in range(1):
        for i, data in enumerate(pii):
            print("epoch: {}, iter {}, img ids: {}, data shape: {}, captions: {}".format(e, i, data[0]["image_ids"][0],
                                                                                         data[0]["images"].shape,
                                                                                         len(data[0]["captions"])))
    total_time = time() - start
    print("time: {}".format(total_time))


if __name__ == '__main__':
    # numpy_test()
    tensor_test()
