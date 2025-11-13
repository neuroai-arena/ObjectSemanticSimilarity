import argparse
import csv
import glob
import re
import h5py
import sys, os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

from scripts.read_cameras import *

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home/fias/postdoc/data/MVImgNet/dataset")
parser.add_argument("--build_hdf5", type=int, default=1)
parser.add_argument("--mode", type=str, default="w")
args=parser.parse_args()





# t=time.time()
# cam = read_cameras_binary("/home/fias/postdoc/data/MVImgNet/263/320097f5/sparse/0/cameras.bin")
# cam = read_cameras_binary("/home/fias/postdoc/data/MVImgNet/34/0200d9f7/sparse/0/cameras.bin")
# cam = read_images_binary("/home/fias/postdoc/data/MVImgNet/34/0200d9f7/sparse/0/images.bin")
# print(time.time()-t)
# print(cam)

#
def convert_to_parquet(name, dataset_schema):
    pd_dataset = pd.read_csv(os.path.join(args.path,name+".csv"))
    table = pa.Table.from_pandas(pd_dataset, schema=dataset_schema)
    pq.write_table(table, os.path.join(args.path, name+".parquet"))



train_csv_file = open(os.path.join(args.path,f"dataset_train_all3.csv"), args.mode)
test_csv_file = open(os.path.join(args.path,f"dataset_test_all3.csv"), args.mode)
val_csv_file = open(os.path.join(args.path,f"dataset_val_all3.csv"), args.mode)
#


csv_writer_train = csv.writer(train_csv_file)
csv_writer_val = csv.writer(val_csv_file)
csv_writer_test = csv.writer(test_csv_file)

if args.mode == "w":
    csv_writer_train.writerow(["path","category","object","frame","index","length", "q0","q1","q2","q3","t0","t1","t2"])
    csv_writer_val.writerow(["path","category","object","frame","index","length", "q0","q1","q2","q3","t0","t1","t2"])
    csv_writer_test.writerow(["path","category","object","frame","index","length", "q0","q1","q2","q3","t0","t1","t2"])
# csv_writer_test.writerow(["path","category","object","frame","index","length"])

dataset_schema = pa.schema([
    ('path', pa.string()),
    ('category', pa.int32()),
    ('object', pa.string()),
    ('frame', pa.int32()),
    ('index', pa.int32()),
    ('length', pa.int32()),
    ('q0', pa.float32()),
    ('q1', pa.float32()),
    ('q2', pa.float32()),
    ('q3', pa.float32()),
    ('t0', pa.float32()),
    ('t1', pa.float32()),
    ('t2', pa.float32()),
])

# dataset_schema_test = pa.schema([
#     ('path', pa.string()),
#     ('category', pa.int32()),
#     ('object', pa.string()),
#     ('frame', pa.int32()),
#     ('index', pa.int32()),
#     ('length', pa.int32())
# ])

if args.build_hdf5:
    hf = h5py.File(os.path.join(args.path, "data_all.h5"), args.mode)



for cat in os.listdir(args.path):
    cat_path = os.path.join(args.path, cat)
    image_names = []
    count_frame = 0
    j = 0
    if not os.path.isdir(cat_path) or cat == "archive":
        continue
    for obj in os.listdir(cat_path):
        obj_path = os.path.join(cat_path, obj)
        imgs_path = os.path.join(obj_path, "images/")

        if not os.listdir(obj_path):
            continue
        camera_file = os.path.join(obj_path, "sparse/0/images.bin")
        if not os.path.isfile(camera_file):
            print("missing camera", cat, obj)
            continue
        cam = read_images_binary(camera_file)
        size_cam, size_dir = len(cam), len([f for f in glob.glob(imgs_path+'*') if re.match("^((?!removed).)*$", f)])
        if size_cam != size_dir:
            print("not all matching images", cat, obj, size_cam, size_dir)
            continue

        # cat_h5_grp = hf.create_group(obj, shape=(len(dataset),), type=h5py.vlen_dtype(np.dtype('uint8')))

        for k, c in sorted(cam.items(), key=lambda item: item[1].name):
            img = c.name
            index = int(img[:3])
            actions = c.qvec.tolist() + c.tvec.tolist()

            path_img = os.path.join(cat, obj,"images", img)
            values = [path_img, cat, obj, count_frame, index, len(cam)] + actions
            if j%10 == 9:
                writer = csv_writer_test
            elif j%10 == 8:
                writer = csv_writer_val
            else:
                writer = csv_writer_train
            writer.writerow(values)
            count_frame += 1
            image_names.append(os.path.join(args.path, path_img))
        j += 1

    if args.build_hdf5:
        cat_dataset = hf.create_dataset(cat, (len(image_names, )), dtype=h5py.vlen_dtype(np.dtype('uint8')))
        for i, img_name in enumerate(image_names):
            cat_dataset[i] = np.fromfile(img_name, dtype=np.uint8)
    print(cat)
train_csv_file.close()
test_csv_file.close()
val_csv_file.close()

convert_to_parquet(f"dataset_train_all3", dataset_schema)
convert_to_parquet(f"dataset_val_all3", dataset_schema)
convert_to_parquet(f"dataset_test_all3", dataset_schema)
