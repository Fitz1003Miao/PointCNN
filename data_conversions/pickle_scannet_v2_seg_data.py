#!/usr/bin/python3
#-*- coding: utf-8 -*-
import pickle
import argparse, os, json
import plyfile
from plyfile import PlyData
import numpy as np

def pickleScenes(scenes_list, outpath, has_labels, label_map, label_list, withrgb):
    pts_list = []
    labels_list = []

    for idx, scene_folder in enumerate(scenes_list):
        scene_name = scene_folder.split('/')[-1]
        
        ply_file = os.path.join(scene_folder, "%s_vh_clean_2.ply" % (scene_name))
        jsonfile = os.path.join(scene_folder, '%s_vh_clean_2.0.010000.segs.json' % (scene_name))
        aggjsonfile = os.path.join(scene_folder, "%s.aggregation.json" % (scene_name))

        print("\n Read ply file:", ply_file)
        plydata = PlyData.read(ply_file).elements[0].data
        pt_num = len(plydata)
        print("points num:", pt_num)
        
        if has_labels:
            
            print("Read json file:", jsonfile)
            json_data = json.load(open(jsonfile))
            segIndices = json_data['segIndices']

            print("Read aggregation json file:", aggjsonfile)
            aggjson_data = json.load(open(aggjsonfile))
            segGroups = aggjson_data['segGroups']

        if withrgb:
            pts = np.zeros([pt_num, 6])
        else:
            pts = np.zeros([pt_num, 3])
        labels = np.zeros([pt_num,], dtype = np.int32)

        for k, pt in enumerate(plydata):
            if withrgb:
                pts[k,:] = np.array([pt[0], pt[1], pt[2], pt[3] * 1.0 / 255, pt[4] * 1.0 / 255, pt[5] * 1.0 / 255]).reshape(1, -1)
            else:
                pts[k,:] = np.array([pt[0], pt[1], pt[2]]).reshape(1, -1)

            if has_labels:
                seg_indice = segIndices[k]
                
                label = ""
                for seg in segGroups:
                    segments = seg['segments']
                    if seg_indice in segments:
                        label = seg['label']
                        break

                if label == "" or label_map[label] not in label_list:
                    labels[k,] = 0
                else:
                    labels[k,] = label_list[label_map[label]]
        
        print(">>>>>>>>>>>> {} / {} -{} complete >>>>>>>>>>>>".format(idx, len(scenes_list), scene_name))
        
        pts_list.append(pts)
        labels_list.append(labels)

    return pts_list, labels_list
        

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("-f", "--folder", help = "Path to whole data")
    parse.add_argument("-b", "--bench", help = "Path to Benchmark")
    parse.add_argument("-o", "--outpath", help = "Path to output")
    parse.add_argument("-r", "--rgb", help = "Flag point with rgb", action = 'store_true')

    args = parse.parse_args()
    print(args)

    folder = args.folder
    benchmark = args.bench
    outpath = args.outpath

    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok = True)

    train_files_txt = os.path.join(benchmark, "scannetv2_train.txt")
    val_files_txt = os.path.join(benchmark, "scannetv2_val.txt")
    test_files_txt = os.path.join(benchmark, "scannetv2_test.txt")

    label_tsv = os.path.join(benchmark, "scannetv2-labels.combined.tsv")
    label_NYUv2 = os.path.join(benchmark, "labelids_all.txt")

    with open(train_files_txt, "r") as f:
        train_scenes_list = [os.path.join(folder, 'train', scene.strip('\n')) for scene in f.readlines()]
        f.close()

    with open(val_files_txt, "r") as f:
        val_scenes_list = [os.path.join(folder, 'train', scene.strip('\n')) for scene in f.readlines()]
        f.close()

    with open(test_files_txt, "r") as f:
        test_scenes_list = [os.path.join(folder, 'test', scene.strip('\n')) for scene in f.readlines()]
        f.close()

    with open(label_tsv, "r") as tsv_f:
        label_map = {}

        for k, line in enumerate(tsv_f.readlines()):
            if k > 0:
                line_s = line.strip().split('\t')
                category = line_s[1]
                NYUv2_category = line_s[7]
                label_map[category] = NYUv2_category
        tsv_f.close()

    with open(label_NYUv2, "r") as label_f:
        label_list = {}
        for k, line in enumerate(label_f.readlines()):
            line_s = line.strip().split('\t')
            label_id = int(line_s[0])
            category = line_s[1]
            label_list[category] = label_id
        label_f.close()

    print("train num is {}, val num is {}, test num is {}".format(len(train_scenes_list), len(val_scenes_list), len(test_scenes_list)))
    pts_list, labels_list = pickleScenes(scenes_list = train_scenes_list, outpath = outpath, has_labels = True, label_map = label_map, label_list = label_list, withrgb = args.rgb)
    
    train_pickle_file = os.path.join(outpath, "scannet_train.pickle")
    f = open(train_pickle_file, "wb")
    pickle.dump(pts_list, f)
    pickle.dump(labels_list, f)
    f.close()
    
    pts_list, labels_list = pickleScenes(scenes_list = val_scenes_list, outpath = outpath, has_labels = True, label_map = label_map, label_list = label_list, withrgb = args.rgb)
    
    val_pickle_file = os.path.join(outpath, "scannet_val.pickle")
    f = open(val_pickle_file, "wb")
    pickle.dump(pts_list, f)
    pickle.dump(labels_list, f)
    f.close()
    
    pts_list, _ = pickleScenes(scenes_list = test_scenes_list, outpath = outpath, has_labels = False, label_map = label_map, label_list = label_list, withrgb = args.rgb)
    
    test_pickle_file = os.path.join(outpath, "scannet_test.pickle")
    f = open(test_pickle_file, "wb")
    pickle.dump(pts_list, f)
    f.close()

if __name__ == "__main__":
    main()