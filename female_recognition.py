from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os

is_evaluate = True

def learn():
    image_list = []
    label_list = []

    # data/train以下のディレクトリを読み込む
    for dir in os.listdir('data/train'):
        dir1 = 'data/train/' + dir
        label = 0

        if dir == 'kaela':
            label = 0
        elif dir == 'maki':
            label = 1
        else:
            label = 2
        
        for file in os.listdir(dir1):
            label_list.append(label)
            filepath = dir1 + '/' + file
            image = np.array(Image.open(filepath).resize((64, 64)))
            print(filepath)

            image = image.transpose(2, 0, 1).reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
            image_list.append(image / 255.)

    # numpy配列に変換
    image_list = np.array(image_list)

    # ラベル配列の変換
    label_list = to_categorical(label_list)

    # モデル生成
    model = Sequential()
    model.add(Dense(256, input_dim=12288))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(3))
    model.add(Activation("softmax"))

    # オプティマイザの設定
    opt = Adam(lr=0.001)
    # モデルをコンパイル
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # 学習を実行。10%はテストに使用。
    model.fit(image_list, label_list, epochs=200, batch_size=100, validation_split=0.1)

    # 学習させたモデルを保存
    model.save('female_recognition_model.h5')

def evaluate():
    
    # 保存してあるモデルの呼び出し
    model = load_model('female_recognition_model.h5')

    label_list = []

    # テスト用ディレクトリの画像でテスト 
    total = 0
    ok_count = 0

    for dir in os.listdir('data/test'):
        dir1 = 'data/test/' + dir
        label = 0

        if dir == 'kaela':
            label = 0
        elif dir == 'maki':
            label = 1
        else:
            label = 2

        for file in os.listdir(dir1):
            label_list.append(label)
            filepath = dir1 + '/' + file
            image = np.array(Image.open(filepath).resize((64, 64)))
            print(filepath)
            image = image.transpose(2, 0, 1).reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype('float32')[0]
            result = model.predict_classes(np.array(np.array([image / 255])))
            print('label:', label, 'result:', result[0])

            total += 1

            if label == result[0]:
                ok_count += 1

    print('correct: ', ok_count / total * 100, '%')

def main():
    if is_evaluate:
        evaluate()
    else:
        learn()

if __name__ == '__main__':
    main()
        
