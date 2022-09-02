import os
from unittest import result
# request フォームから送信した情報を扱うためのモジュール
# redirect  ページの移動
# url_for アドレス遷移
from flask import Flask, request, redirect, url_for, flash, render_template
# ファイル名をチェックする関数
from werkzeug.utils import secure_filename
# 動画のダウンロード
from flask import send_from_directory

import argparse
import logging
import time
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


# ------------------- 初期設定 --------------------------
# 動画のアップロード先のディレクトリ
UPLOAD_FOLDER = '/Users/zouzatakumi/Desktop/Flask_app/uploads/'
# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['mp4'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ----------------- 拡張子の確認 ------------------------
def allowed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ファイルを受け取る方法の指定
@app.route('/', methods=['GET', 'POST'])
def uploads_file():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        # データの取り出し
        file = request.files['file']
        # ファイル名がなかった時の処理
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)

        # アップロードされたファイルの保存
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # ーーーーーーーーーーーーーーーーーーーーー OpenPoseの実行 ーーーーーーーーーーーーーーーーーーーーーーー
        logger = logging.getLogger('TfPoseEstimator-Video')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        fps_time = 0

        #parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
        #parser.add_argument('--video', type=str, default=file.filename)
        
        #parser.add_argument('--write_video', type=str, default=outfile[0] + ".mod." + outfile[1])
        #parser.add_argument('--resize', type=str, default='432x368')
        #parser.add_argument('--resize-out-ratio', type=float, default=4.0)
        #parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
        #parser.add_argument('--show-process', type=bool, default=False)
        #parser.add_argument('--showBG', type=bool, default=True)
        #args = parser.parse_args()
        logger.debug('initialization %s : %s' % ('mobilenet_thin', get_graph_path('mobilenet_thin')))

        # model='mobilenet_thin, size=432, 368
        w = 432
        h = 368
        e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))
        
        # アップロードされたファイルのキャプチャ
        logger.debug('Filename : %s' % ( app.config['UPLOAD_FOLDER'] + file.filename) )
        outfile = file.filename.rsplit(".")
        logger.debug('outfile : %s' % (app.config['UPLOAD_FOLDER'] + outfile[0] + ".mod." + outfile[1]))
        cap = cv2.VideoCapture(app.config['UPLOAD_FOLDER'] + file.filename)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        #fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fmt = cv2.VideoWriter_fourcc('H', '2', '6', '4')
        
        # 解析結果の表示
        @app.route('./uploads')
        def index():
            data = ？
            return render_template('？', result=data)

        # 出力ファイル
        outfile = file.filename.rsplit(".")
        writer = cv2.VideoWriter(app.config['UPLOAD_FOLDER'] + outfile[0] + ".mod." + outfile[1], fmt, fps, (width, height))
    
        with open('humans.txt', mode='w') as f:
            if cap.isOpened() is False:
                print("Error opening video stream or file")

            while cap.isOpened():
                ret_val, image = cap.read()
                if ret_val:
                    logger.debug('image process+')
                    humans = (e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0))
                    #print(humans)
                    #print(type(humans))
                    #print(len(humans))
                    #exit(0)
                    tmp_h = []
                    for human in humans:
                        tmp_h.append(str(human))
                    str_h = ";".join(tmp_h)
                    print(str_h)
                    f.write(str_h)
                    f.write("\n")
                    #exit(0)
                    #if not args.showBG:
                    #image = np.zeros(image.shape)
            
                    logger.debug('postprocess+')
                    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            
                    logger.debug('show+')
                    cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    #cv2.imshow('tf-pose-estimation result', image)
                    #if args.write_video:
                    writer.write(image)

                    fps_time = time.time()
                    if cv2.waitKey(1) == 27:
                        break
                else:
                    break
        #cv2.destroyAllWindows()
        logger.debug('finished+')
        
        # ファイルのチェック
        #if file and allwed_file(file.filename):
            # 危険な文字を削除（サニタイズ処理）
            #filename = secure_filename(file.filename)
            # ファイルの保存
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # アップロード後のページに転送
            #return redirect(url_for('uploaded_file', filename=filename))
        #解析後のファイルにリダイレクト
        return redirect(url_for('uploaded_file', filename=outfile[0] + ".mod." + outfile[1]))
    return '''
    <!doctype html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>
                ファイルをアップロードして判定しよう
            </title>
        </head>
        <body>
            <h1>
                ファイルをアップロードして判定しよう
            </h1>
            <form method = post enctype = multipart/form-data>
            <p><input type=file name = file>
            <input type = submit value = Upload>
            </form>
            <div>
                <h2>解析結果</h2>
                <ul>
                    {% for results in result %}
                        <li><a>{{ results }}</a></li>
                    {% endfor %}
                </ul>
            </div>
        </body>
'''

@app.route('/uploads/<filename>')
# ファイルを表示する
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.debug = True
    app.run()
    
    
    