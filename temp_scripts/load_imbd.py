# -*- coding: utf-8 -*- 
import lmdb
import msgpack
import os
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt



def pil_from_raw_rgb(raw):
    byte_raw = BytesIO(raw)
    return Image.open(byte_raw).convert('RGB')
size = 320*240
root = "/media/gandalf/AE3416073415D2E7/ucf_flow/ucf101_flow/flow/UCF101"
path = os.path.join(root, 'ucf101_frame.lmdb')

env = lmdb.open(path, subdir=os.path.isdir(path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)



with env.begin(write=False) as txn:
            db_length = msgpack.loads(txn.get(b'__len__'))
            db_keys = msgpack.loads(txn.get(b'__keys__'))
            db_order = msgpack.loads(txn.get(b'__order__'))
get_video_id = dict(
    zip([i for i in db_order], ['%09d' % i for i in range(len(db_order))]))

#with env.begin(write=False) as txn:
#            raw = msgpack.loads(txn.get(get_video_id['000005357'].encode('ascii')))
txn = env.begin(buffers=True)
buf = txn.get('000005357'.encode('ascii'))
buf_bytes = bytes(buf)
nparr = np.fromstring(buf_bytes, np.uint8)
im = pil_from_raw_rgb(buf_bytes)
nparr = np.fromstring(buf_bytes, np.uint8)
video = nparr[0:1920000].reshape(320, 240, 25)
print('done')