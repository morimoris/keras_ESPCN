import model
import data_create
import argparse
import os
import cv2

import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    
    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, 1, name=None)

    train_height = 17
    train_width = 17
    test_height = 200
    test_width = 200

    mag = 3.0
    cut_traindata_num = 10
    cut_testdata_num = 1

    train_file_path = "./train_data" #写真が入ったフォルダ
    test_file_path = "./test_data" #写真が入ったフォルダ

    BATSH_SIZE = 256
    EPOCHS = 1000
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='espcn', help='espcn, evaluate')

    args = parser.parse_args()

    if args.mode == "espcn":
        train_x, train_y = data_create.save_frame(train_file_path,   #切り取る画像のpath
                                                cut_traindata_num,  #データセットの生成数
                                                train_height, #保存サイズ
                                                train_width,
                                                mag)   #倍率
                                                
        model = model.ESPCN(mag) 
        model.compile(loss = "mean_squared_error",
                        optimizer = opt,
                        metrics = [psnr])
#https://keras.io/ja/getting-started/faq/
        model.fit(train_x,
                    train_y,
                    epochs = EPOCHS)

        model.save("espcn_model.h5")

    elif args.mode == "evaluate":
        path = "espcn_model"
        exp = ".h5"
        new_model = tf.keras.models.load_model(path + exp, custom_objects={'psnr':psnr})

        new_model.summary()

        test_x, test_y = data_create.save_frame(test_file_path,   #切り取る画像のpath
                                                cut_testdata_num,  #データセットの生成数
                                                test_height, #保存サイズ
                                                test_width,
                                                mag)   #倍率
        print(len(test_x))
        pred = new_model.predict(test_x)
        path = "result_epo_" + epo
        os.makedirs(path, exist_ok = True)
        path = path + "/"

        for i in range(10):
            ps = psnr(tf.reshape(test_y[i], [test_height, test_width, 1]), pred[i])
            print("psnr:{}".format(ps))

            before_res = tf.keras.preprocessing.image.array_to_img(tf.reshape(test_x[i], [int(test_height / mag), int(test_width / mag), 1]))
            change_res = tf.keras.preprocessing.image.array_to_img(tf.reshape(test_y[i], [test_height, test_width, 1]))
            y_pred = tf.keras.preprocessing.image.array_to_img(pred[i])

            before_res.save(path + "low_" + str(i) + ".jpg")
            change_res.save(path + "high_" + str(i) + ".jpg")
            y_pred.save(path + "pred_" + str(i) + ".jpg")

    else:
        raise Exception("Unknow --mode")
 
   
