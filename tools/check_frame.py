if __name__ == '__main__':
    txt_path = 'YOLOX_outputs/yolox_x_ablation/track_results/MOT17-09-FRCNN.txt'
    low_score_frame = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            linelist = line.split(',')

            img_id = linelist[0]
            obj_id = linelist[1]
            obj_score = linelist[6]
            if float(obj_score) < 0.6:
                low_score_frame.append(img_id)
    print(low_score_frame)
