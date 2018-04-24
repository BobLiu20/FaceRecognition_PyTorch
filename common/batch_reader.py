#coding=utf-8
import os
import sys
import numpy as np
import cv2
import math
import signal
import random
import time
#  using torch multiprocessing instead of python
import torch
from torch.multiprocessing import Process, Queue, Event
from torch.autograd import Variable

exitEvent = Event() # for noitfy all process exit.

def handler(sig_num, stack_frame):
    global exitEvent
    exitEvent.set()
signal.signal(signal.SIGINT, handler)

def preprocess_func(batch):
    datas = np.asarray(batch[0], dtype=float)
    datas = torch.from_numpy(datas).float()
    datas = datas.permute(0, 3, 1, 2)
    datas -= 127.5
    datas *= 0.0078125
    labels = torch.from_numpy(batch[1]).long()
    return Variable(datas, requires_grad=False), Variable(labels, requires_grad=False)

class BatchReader():
    def __init__(self, **kwargs):
        # param
        self._kwargs = kwargs
        self._batch_size = kwargs['batch_size']
        self._process_num = kwargs['process_num']
        self._img_size = kwargs['img_size']
        self._debug = kwargs['debug']
        # total lsit
        self._sample_list = [] # each item: (filepath, landmarks, ...)
        self._total_sample = 0
        # real time buffer
        self._process_list = []
        self._output_queue = []
        for i in range(self._process_num):
            self._output_queue.append(Queue(maxsize=1)) # for each process
        # epoch
        self._idx_in_epoch = 0
        self._curr_epoch = 0
        self._max_epoch = kwargs['max_epoch']
        # start buffering
        self._start_buffering(kwargs['input_paths'])

    def batch_generator(self):
        __curr_queue = 0
        while True:
            self.__update_epoch()
            while True:
                __curr_queue += 1
                if __curr_queue >= self._process_num:
                    __curr_queue = 0
                try:
                    datas = self._output_queue[__curr_queue].get(block=True, timeout=0.01)
                    break
                except Exception as ex:
                    pass
            yield datas

    def get_epoch(self):
        return self._curr_epoch

    def should_stop(self):
        if exitEvent.is_set() or self._curr_epoch > self._max_epoch:
            exitEvent.set()
            self.__clear_and_exit()
            return True
        else:
            return False

    def __clear_and_exit(self):
        print ("[Exiting] Clear all queue.")
        while True:
            time.sleep(1)
            _alive = False
            for i in range(self._process_num):
                try:
                    self._output_queue[i].get(block=True, timeout=0.01)
                    _alive = True
                except Exception as ex:
                    pass
            if _alive == False: break
        print ("[Exiting] Confirm all process is exited.")
        for i in range(self._process_num):
            if self._process_list[i].is_alive():
                print ("[Exiting] Force to terminate process %d"%(i))
                self._process_list[i].terminate()
        print ("[Exiting] Batch reader clear done!")

    def _start_buffering(self, input_paths):
        print ("loading data set...")
        if type(input_paths) in [str, unicode]:
            input_paths = [input_paths]
        count = 0
        for input_path in input_paths:
            for line in open(input_path):
                if self._debug:
                    if count > 10000:
                        break
                    count += 1
                # parse line
                idx = line.rfind(' ')
                _path = line[:idx]
                _id = int(line[idx+1:].strip('\n'))
                self._sample_list.append([_path, _id])
        self._total_sample = len(self._sample_list)
        num_per_process = int(math.ceil(self._total_sample / float(self._process_num)))
        for idx, offset in enumerate(range(0, self._total_sample, num_per_process)):
            p = Process(target=self._process, args=(idx, self._sample_list[offset: offset+num_per_process]))
            p.start()
            self._process_list.append(p)
        print ("load done.")

    def _process(self, idx, sample_list):
        sample_cnt = 0 # count for one batch
        image_list, label_list = [], [] # one batch list
        while True:
            for sample in sample_list:
                # preprocess
                image = cv2.imread(sample[0])
                if image is None:
                    print ("Bad image: {}".format(sample[0]))
                    print ("Batch reader exit now.")
                    global exitEvent
                    exitEvent.set()
                    sys.exit()
                image = cv2.resize(image, (self._img_size, self._img_size))
                if np.random.randint(0, 2) == 0:
                    image = np.fliplr(image)
                # sent a batch
                sample_cnt += 1
                image_list.append(image)
                label_list.append(sample[1])
                if sample_cnt >= self._kwargs['batch_size']:
                    datas = (np.array(image_list), np.array(label_list))
                    datas = preprocess_func(datas)
                    self._output_queue[idx].put(datas)
                    sample_cnt = 0
                    image_list, label_list = [], []
                # if exit
                if exitEvent.is_set():
                    break
            if exitEvent.is_set():
                break
            np.random.shuffle(sample_list)

    def __update_epoch(self):
        self._idx_in_epoch += self._batch_size
        if self._idx_in_epoch > self._total_sample:
            self._curr_epoch += 1
            self._idx_in_epoch = 0

# use for unit test
if __name__ == '__main__':
    kwargs = {
        'input_paths': "/world/data-c9/liubofang/training/"\
                       "face_recognition/train_tupu_696877_caffe.lst",
        'batch_size': 64,
        'process_num': 2,
        'img_size': 112,
        'max_epoch':1,
    }
    b = BatchReader(**kwargs)
    g = b.batch_generator()
    output_folder = "output_tmp/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    import time
    start_time = time.time()
    while not b.should_stop():
        end_time = time.time()
        print ("get new batch...epoch: %d. cost: %.3f"%(
                b.get_epoch(), end_time-start_time))
        start_time = end_time
        datas = g.next()
        for idx, image in enumerate(datas[0]):
            label = datas[1][idx]
            if idx > 20: # only see first 10
                break
            cv2.imwrite("%s/%d_%d.jpg"%(output_folder, idx, label), image)
        break

