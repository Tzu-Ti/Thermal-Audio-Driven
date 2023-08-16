import glob, os
import subprocess
import argparse
import threading
import cv2
from tqdm import tqdm

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='LRS2')
    parser.add_argument('--N_thread', type=int, default=1)
    parser.add_argument('--video2facepng', action="store_true")
    parser.add_argument('--mp42wav', action="store_true")
    return parser.parse_args()

def split_n_fold(lst, N):
    length = len(lst)
    fold_length = length // N
    start = 0
    end = fold_length
    n_fold = []
    for n in range(N):
        if n == N-1: # last loop
            end = -1
        n_fold.append(lst[start: end])
        start = end
        end = end + fold_length
    return n_fold

def job_mp42wav(lst):
    for path in tqdm(lst):
        new_path = path.replace('.mp4', '.wav')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(path, new_path)
        subprocess.call(command, shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT)
import shutil
import face_alignment
import torch
def job_video2facepng(lst):
    def detect_face(img):
        bboxes = fa.face_detector.detect_from_image(img)
        if len(bboxes) == 0:
            return img
        bbox = bboxes[0]
        bbox = [int(i) for i in bbox]
        x1, y1, x2, y2, score = bbox
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        face = img[y1: y2, x1: x2, :]
        return face
    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')
    for path in tqdm(lst):
        save_folder = path.split('.')[0]
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        video_stream = cv2.VideoCapture(path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        assert fps == 25, 'FPS is not 25'
        i = 0
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            face = detect_face(frame)
            new_path = os.path.join(save_folder, "{:03d}.png".format(i))
            cv2.imwrite(new_path, face)
            i += 1

def main():
    args = parse()
    if args.dataset == 'LRW':
        video_paths = glob.glob("/root/LRW/lipread_mp4/*/train/*.mp4")
    elif args.dataset == 'LRS2':
        video_paths = glob.glob("/root/LRS2/mvlrs_v1/main/*/*.mp4")

    n_fold = split_n_fold(lst=video_paths, N=args.N_thread)
    if args.mp42wav:
        job = job_mp42wav
    elif args.video2facepng:
        job = job_video2facepng
    threads = []
    for i in range(args.N_thread):
        threads.append(threading.Thread(target=job, args=(n_fold[i], )))
        threads[i].start()

    for i in range(args.N_thread):
        threads[i].join()

if __name__ == '__main__':
    main()