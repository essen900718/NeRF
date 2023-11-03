import os
import tensorflow as tf
import numpy as np
import imageio 
import json




trans_t = lambda t : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=tf.float32)

rot_phi = lambda phi : tf.convert_to_tensor([
    [1,0,0,0],
    [0,tf.cos(phi),-tf.sin(phi),0],
    [0,tf.sin(phi), tf.cos(phi),0],
    [0,0,0,1],
], dtype=tf.float32)

rot_theta = lambda th : tf.convert_to_tensor([
    [tf.cos(th),0,-tf.sin(th),0],
    [0,1,0,0],
    [tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=tf.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w
    


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits: # 所有圖片的位姿都存在 transforms_{}.json 裡
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []  # 存所有圖片
    all_poses = [] # 存所有位姿
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix'])) # 一個4*4的矩陣 如何把相機視角轉換到世界視角
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA) # 把圖片 Normalize 保持四個通道因為要學習density
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)] # 紀錄每個split有多少張圖片 [100 100 200] > [0~99 100~199 200~399]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2] # 800*800
    camera_angle_x = float(meta['camera_angle_x']) # 相機的視角大小 (FOV)
    focal = .5 * W / np.tan(.5 * camera_angle_x) # 計算焦距大小
    
    render_poses = tf.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)
    
    if half_res: # 要不要用一半的解析度訓練
        imgs = tf.image.resize_area(imgs, [400, 400]).numpy() # 因為BLENDER是800*800 因此一半即是400*400
        H = H//2
        W = W//2
        focal = focal/2. # 圖片大小減半 焦距也要減半
        
    return imgs, poses, render_poses, [H, W, focal], i_split


