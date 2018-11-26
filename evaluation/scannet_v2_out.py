#!/usr/bin/python3
"""Merge blocks and evaluate scannet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import plyfile
import numpy as np
import argparse
import h5py
import pickle

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datafolder', '-d', help='Path to input *_pred.h5', required=True)
    parser.add_argument('--picklefile', '-p', help='Path to scannet_test.pickle', required=True)
    parser.add_argument('--file', '-f', help = "Path to contain filename file .txt", required = True)
    parser.add_argument('--outpath', '-o', help = "Path to output folder", required = True)

    args = parser.parse_args()
    print(args)

    filenamelist = [line.strip('\n') for line in open(args.file, 'r').readlines()]
    
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath, exist_ok = True)

    file_list = os.listdir(args.datafolder)
    pred_list = [pred for pred in file_list if pred.split(".")[-1] == "h5" and "pred" in pred]
    
    #load scannet_test.pickle file
    file_pickle = open(args.picklefile, 'rb')
    xyz_all = pickle.load(file_pickle, encoding='latin1') # encoding keyword for python3
    file_pickle.close()

    pickle_dict = {}
    for room_idx, xyz in enumerate(xyz_all):

        room_pt_num = xyz.shape[0]
        room_dict = {}

        room_dict["merged_label_zero"] = np.zeros((room_pt_num),dtype=int)
        room_dict["merged_confidence_zero"] = np.zeros((room_pt_num),dtype=float)
        room_dict["merged_label_half"] = np.zeros((room_pt_num), dtype=int)
        room_dict["merged_confidence_half"] = np.zeros((room_pt_num), dtype=float)
        room_dict["final_label"] = np.zeros((room_pt_num), dtype=int)

        pickle_dict[room_idx] = room_dict

    # load block preds and merge them to room scene
    for pred_file in pred_list:

        print("process:", os.path.join(args.datafolder, pred_file))
        test_file = pred_file.replace("_pred","")

        # load pred .h5
        data_pred = h5py.File(os.path.join(args.datafolder, pred_file))

        pred_labels_seg = data_pred['label_seg'][...].astype(np.int64)
        pred_indices = data_pred['indices_split_to_full'][...].astype(np.int64)
        pred_confidence = data_pred['confidence'][...].astype(np.float32)
        pred_data_num = data_pred['data_num'][...].astype(np.int64)

        
        if 'zero' in pred_file:
            for b_id in range(pred_labels_seg.shape[0]):
                indices_b = pred_indices[b_id]
                for p_id in range(pred_data_num[b_id]):
                    room_indices = indices_b[p_id][0]
                    inroom_indices = indices_b[p_id][1]
                    pickle_dict[room_indices]["merged_label_zero"][inroom_indices] = pred_labels_seg[b_id][p_id]
                    pickle_dict[room_indices]["merged_confidence_zero"][inroom_indices] = pred_confidence[b_id][p_id]
        else:
            for b_id in range(pred_labels_seg.shape[0]):
                indices_b = pred_indices[b_id]
                for p_id in range(pred_data_num[b_id]):
                    room_indices = indices_b[p_id][0]
                    inroom_indices = indices_b[p_id][1]
                    pickle_dict[room_indices]["merged_label_half"][inroom_indices] = pred_labels_seg[b_id][p_id]
                    pickle_dict[room_indices]["merged_confidence_half"][inroom_indices] = pred_confidence[b_id][p_id]

    for room_id in pickle_dict.keys():

        final_label = pickle_dict[room_id]["final_label"]
        merged_label_zero = pickle_dict[room_id]["merged_label_zero"]
        merged_label_half = pickle_dict[room_id]["merged_label_half"]
        merged_confidence_zero = pickle_dict[room_id]["merged_confidence_zero"]
        merged_confidence_half = pickle_dict[room_id]["merged_confidence_half"]

        final_label[merged_confidence_zero >= merged_confidence_half] = merged_label_zero[merged_confidence_zero >= merged_confidence_half]
        final_label[merged_confidence_zero < merged_confidence_half] = merged_label_half[merged_confidence_zero < merged_confidence_half]

        filepath = os.path.join(args.outpath, filenamelist[room_id] + '.txt')
        
        np.savetxt(filepath, final_label, fmt = '%d')

if __name__ == '__main__':
    main()
