import numpy as np 
import os 
import shutil 
import cv2
import glob
import random
seed = 228
np.random.seed(seed)
random.seed(seed)

def move(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


class MCH_dataset():
    def __init__(self,path_root):
        #self.is_binary = True
        self.is_255 = True
        self.path_root = path_root
        
        # self.dataset_name = [
        # 'bottle', 'cable', 'capsule', 'carpet', 'grid',
        # 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        # 'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
        # ]
        
        self.dataset_name = ['3D','Corner','Hole','RT','SJ']
        #self.dataset_name = ["pill"]
    def Binary(self,mask):
        if self.is_255:
            mask[mask<=128] = 0
            mask[mask>128] = 1
        else:
            mask[mask<=0] = 0
            mask[mask>0] = 1
        return mask
    
    def make_dirs(self,des_root):
        for data_name in self.dataset_name:
            name = data_name
            dir_list = [os.path.join(des_root,name,"train","good"),os.path.join(des_root,name,"test","good"),os.path.join(des_root,name,"test","anomaly"),os.path.join(des_root,name,"ground_truth","anomaly")]
            for dir in dir_list:
                if not os.path.exists(dir):
                    os.makedirs(dir)

    # train  test  ground_truth
    def make_VAND(self, binary, to_255, des_path_root, id):
        def tail3_from_stem(stem: str, fallback_id: int) -> str:
            parts = stem.split('-')
            if len(parts) >= 3:
                return "_".join(parts[-3:])
            else:
                return "____"
            
        self.make_dirs(des_path_root)
        for data_name in self.dataset_name:
            print("Processing :{}".format(data_name))
            data_path = os.path.join(self.path_root, data_name)

            neg_dir = os.path.join(data_path, "neg")
            pos_dir = os.path.join(data_path, "pos")

            dst_train_good = os.path.join(des_path_root, data_name, "train", "good")
            dst_test_anom = os.path.join(des_path_root, data_name, "test", "anomaly")
            dst_gt_anom = os.path.join(des_path_root, data_name, "ground_truth", "anomaly")

            os.makedirs(dst_train_good, exist_ok=True)
            os.makedirs(dst_test_anom, exist_ok=True)
            os.makedirs(dst_gt_anom, exist_ok=True)

            if os.path.isdir(neg_dir):
                neg_imgs = sorted(glob.glob(os.path.join(neg_dir, "*.jpg")))
                for img_path in neg_imgs:
                    raw_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if raw_img is None:
                        continue
                    stem = os.path.splitext(os.path.basename(img_path))[0]
                    tail3 = tail3_from_stem(stem, id) # e.g., glass-null-008836
                    save_img_path = os.path.join(dst_train_good, f"MCH_f{tail3}_{str(id).zfill(6)}.png")
                    cv2.imwrite(save_img_path, raw_img)
                    id += 1

            if os.path.isdir(pos_dir):
                pos_imgs = sorted(glob.glob(os.path.join(pos_dir, "*.jpg")))
                for img_path in pos_imgs:
                    stem = os.path.splitext(os.path.basename(img_path))[0]
                    mask_path = os.path.join(pos_dir, stem + ".png")

                    raw_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if raw_img is None:
                        continue

                    if not os.path.exists(mask_path):
                        print(f"[WARN] mask missing for: {img_path}")
                        continue

                    raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if raw_mask is None:
                        continue

                    # 可选：二值化 & 转255
                    if binary:
                        raw_mask = self.Binary(raw_mask.copy())
                    if to_255:
                        raw_mask = (raw_mask > 0).astype(np.uint8) * 255

                    # 保存
                    tail3 = tail3_from_stem(stem, id)
                    save_img_path = os.path.join(dst_test_anom, f"MCH_anomaly_f{tail3}_{str(id).zfill(6)}.png")
                    save_mask_path = os.path.join(dst_gt_anom, f"MCH_anomaly_f{tail3}_{str(id).zfill(6)}.png")
                    cv2.imwrite(save_img_path, raw_img)
                    cv2.imwrite(save_mask_path, raw_mask)
                    id += 1

        print("MCH finished !")
        return id
