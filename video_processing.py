
from skimage.metrics import structural_similarity as compare_ssim
from collections import Counter
import numpy as np
import tensorflow as tf
from PIL import Image
import ffmpeg, shutil, glob
import multiprocessing as mp
import pandas as pd
import re
# import nest_asyncio, uvicorn, os, pathlib
import yaml
import cv2#, wandb
import os
import pathlib


def split_video_frames(video_name, extension, source_folder, dest_folder):
    print("Checking if video already split", end="\r")
    path = f'{dest_folder}/frame00001.png'
    if os.path.exists(path):
        text_output = f'Video: {video_name} already split. Skipping split.'
        print(text_output)
        return
    os.mkdir(str(dest_folder), exist_ok=True)

    video_location = f'{source_folder}/{video_name}.{extension}'
    video_capture = cv2.VideoCapture(video_location)
    saved_frame_name = 1

    while True:
        print("Frame: " + format(saved_frame_name, '05d'), end="\r")
        success, frame = video_capture.read()

        if success:
            cv2.imwrite(f"{str(dest_folder)}/frame{format(saved_frame_name, '05d')}.png", frame)
            saved_frame_name += 1
        else:
            break
    print("Split video frames")

def parallel_laplacian_variance(file):
    print(file + "             ", end="\r")
    img = cv2.imread(file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return laplacian


def remove_blurry_images(folder_name, extension, delete_frames=False):
    files = sorted(glob.glob(f'{folder_name}/*.{extension}'))
    # blurriness = np.zeros(len(files))
    print("Calculating Average Blurriness")

    # Parallelize
    # blurriness = [pool.apply(parallel_laplacian_variance, args=(file,)) for file in files]
    # pool.close()
    blurriness = [parallel_laplacian_variance(file) for file in files]
    median_blur = float(np.median(blurriness))
    min_blur = float(np.min(blurriness))
    max_blur = float(np.max(blurriness))
    print(median_blur)
    # wandb.log({'Individual Laplacian': blurriness, 'Batch Median Laplacian': median_blur})
    print("Median Blur (Laplacian Variance): " + str(median_blur))
    blur_cutoff = median_blur * 0.95  # + ((1-average_blur)*0.1)
    print("Blur Cutoff (Laplacian Variance): " + str(blur_cutoff))

    print("Removing Noisy Images")

    count = 0
    removed_files = []

    cutoffs = {}
    for percentage in np.arange(0, 1, 0.05):
        cutoffs.update({f'{percentage}': float(np.quantile(blurriness, percentage))})

    remainders = {}
    number_of_original_files = len(files)
    for j in np.arange(300, 1000, 100):
        remainder = sum(i > j for i in blurriness)
        remainders[f'{j} Threshold'] = float(remainder / number_of_original_files)

    remainders = dict(remainders)

    for i in range(len(files)):
        if blurriness[i] < blur_cutoff:
            # print("Deleting " + files[i] + " - Laplacian Noisiness: " + str(blurriness[i]))
            removed_files.append(files[i])
            if delete_frames == True:
                os.remove(files[i])
            count += 1
    blur_ratio = count / len(files)
    # wandb.log({'Noisy Frame Ratio': blur_ratio})
    print(f"Done Checking Frames, {count} frames removed.                 ")
    return {'Laplacian Threshold Remainders': remainders, 'Total Original Frames': number_of_original_files,
            'Removed Blurry Frame Count': count, 'Removed Blurry Frames': removed_files,
            'Median Laplacian Variance': median_blur, 'Minimum Laplacian Variance': min_blur,
            'Maximum Laplacian Variance': max_blur, 'Noisy Frame Ratio': blur_ratio, 'Laplacian Cutoffs': cutoffs}

def compare_images(i, files):
    image1 = cv2.imread(files[i])
    image2 = cv2.imread(files[i + 1])
    image_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    try:
        diff, _ = compare_ssim(image_gray1, image_gray2, full=True)
    except:
        image_gray2 = cv2.resize(image_gray2, image_gray1.shape, interpolation=cv2.INTER_AREA)
        diff, _ = compare_ssim(image_gray1, image_gray2, full=True)
    return diff

def remove_duplicates(folder_name, extension, delete_frames=False):
    files = sorted(glob.glob(f'{folder_name}/*.{extension}'))
    print("Removing Duplicate and Highly Similar Frames\nCalculating Frame Similarities")
    # diff = np.zeros(len(files)-1)

    # Parallelize
    # pool = mp.Pool(mp.cpu_count())

    diff = [compare_images(i, files) for i in range(len(files) - 1)]

    # pool.close()



    median_diff = float(np.median(diff))
    # wandb.log({'Individual Frame Similarities': diff, 'Batch Median Frame Similarity': median_diff})

    diff_cutoff = median_diff * 1.05

    if diff_cutoff < 0.95:
        diff_cutoff = 0.95

    print(f'Similarity Cutoff (OpenCV Compare Images): {diff_cutoff}')
    print('Removing Duplicate Images')

    count = 0
    removed_files = []

    cutoffs = {}
    for percentage in np.arange(0, 1, 0.05):
        cutoffs.update({f'{percentage}': float(np.quantile(diff, percentage))})

    remainders = {}
    number_of_original_files = len(files)
    for j in np.arange(0.2, 0.8, 0.05):
        remainder = sum(i > j for i in diff)
        remainders[f'{j} Threshold'] = float(remainder / number_of_original_files)

    remainders = dict(remainders)

    for i in range(len(diff)):
        if diff[i] > 0.99:
            # print("Deleting " + files[i] + " - Similarity: " + str(diff[i]), end="\r")
            removed_files.append(files[i])
            if delete_frames == True:
                os.remove(files[i])
            # wandb.log({'Duplicates Similarity': diff})
            count += 1

    duplicate_ratio = count / len(files)
    # wandb.log({'Batch Duplicate Remove Ratio': duplicate_ratio})
    print("Done Checking Frames, " + str(count) + " frames removed.")
    return {'SSIM Threshold Remainders': remainders, 'Removed Duplicate Frame Count': count,
            'Removed Duplicate Frames': removed_files, 'Median Frame Similarity': median_diff,
            'Duplicate Frame Ratio': duplicate_ratio, 'Similarity Cutoffs': cutoffs}