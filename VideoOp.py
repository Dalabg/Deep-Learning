import cv2
import os
import json
import copy
import numpy as np
from tqdm import tqdm
import time
from MeshFlow import motion_propagate
from MeshFlow import mesh_warp_frame
from MeshFlow import generate_vertex_profiles
from Optimization import real_time_optimize_path
PIXELS=16.0

# class Video(object):
#     def __init__(self,cap,frame_rate,frame_width,frame_height,frame_count):
#         self.cap = cap
#         self.frame_rate = frame_rate
#         self.frame_width = frame_width
#         self.frame_height = frame_height
#         self.frame_count = frame_count
#     def GetParameter(self):
#         self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#         self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
#         self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


#  获取变量名函数
def var_name(var,all_var=locals()):
    return [var_name for var_name in all_var if all_var[var_name] is var][0]

def measure_time(method):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        print(method.__name__+' has taken: '+str(end_time-start_time)+' sec')
        return result
    return timed


def GetParameter(cap):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return [frame_rate,frame_width,frame_height,frame_count]

#  操作1：读取路径
@measure_time
def PathRead(cap):
    """
    @param: cap is the cv2.VideoCapture object that is
            instantiated with given video

    Returns:
            returns mesh vertex motion vectors &
            mesh vertex profiles
    """
    # ShiTomasi角特征检测参数 1994论文 用于确定特征点以及度量帧变化大小
    feature_params = dict( maxCorners = 1000,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # lucas kanade光流法参数
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    # 获取第0帧信息，作为基准帧跟踪帧变化
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 初始化x，y方向的运动网格值
    x_motion_meshes = []; y_motion_meshes = []

    # 路径参数
    x_paths = np.zeros((int(old_frame.shape[0]/PIXELS), int(old_frame.shape[1]/PIXELS), 1))
    y_paths = np.zeros((int(old_frame.shape[0]/PIXELS), int(old_frame.shape[1]/PIXELS), 1))

    frame_num = 1
    bar = tqdm(total=frame_count)
    while frame_num < frame_count:

        # 处理帧
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算角部特征
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # 计算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # 选取较好的特征点
        good_new = p1[st==1]
        good_old = p0[st==1]

        # 基于上一帧估计运动网络
        x_motion_mesh, y_motion_mesh = motion_propagate(good_old, good_new, frame)
        try:
            x_motion_meshes = np.concatenate((x_motion_meshes, np.expand_dims(x_motion_mesh, axis=2)), axis=2)
            y_motion_meshes = np.concatenate((y_motion_meshes, np.expand_dims(y_motion_mesh, axis=2)), axis=2)
        except:
            x_motion_meshes = np.expand_dims(x_motion_mesh, axis=2)
            y_motion_meshes = np.expand_dims(y_motion_mesh, axis=2)

        # 生成顶点轮廓
        x_paths, y_paths = generate_vertex_profiles(x_paths, y_paths, x_motion_mesh, y_motion_mesh)

        # 更新帧
        bar.update(1)
        frame_num += 1
        old_frame = frame.copy()
        old_gray = frame_gray.copy()

    bar.close()
    print('已计算各光流路径')
    return [x_motion_meshes, y_motion_meshes, x_paths, y_paths]


#  操作2：稳定路径计算
@measure_time
def stabilize(x_paths, y_paths):
    """
    @param: x_paths is motion vector accumulation on
            mesh vertices in x-direction
    @param: y_paths is motion vector accumulation on
            mesh vertices in y-direction

    Returns:
            returns optimized mesh vertex profiles in
            x-direction & y-direction
    """

    # optimize for smooth vertex profiles
    sx_paths = real_time_optimize_path(x_paths)
    sy_paths = real_time_optimize_path(y_paths)
    return [sx_paths, sy_paths]

def WriteInJson(filepath,dataname,data):
    with open(filepath, 'w') as f:
        json.dump(data.tolist(), f)
        print('已写入' + filepath + '文件', '数据类型为：', type(data))

#  路径写入json函数,即存储中间路径变量
def PathIntoJson(videoname,cap):
    # 传播运动矢量并生成顶点轮廓
    x_motion_meshes, y_motion_meshes, x_paths, y_paths = PathRead(cap)

    # 稳定顶点轮廓
    sx_paths, sy_paths = stabilize(x_paths, y_paths)

    # 获取更新的网格扭曲
    x_motion_meshes = np.concatenate((x_motion_meshes, np.expand_dims(x_motion_meshes[:, :, -1], axis=2)), axis=2)
    y_motion_meshes = np.concatenate((y_motion_meshes, np.expand_dims(y_motion_meshes[:, :, -1], axis=2)), axis=2)
    new_x_motion_meshes = sx_paths - x_paths
    new_y_motion_meshes = sy_paths - y_paths

    text = [x_motion_meshes, y_motion_meshes, new_x_motion_meshes, new_y_motion_meshes]
    path = './TempData/' + videoname +'/'
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
    filepath = path + 'x_motion_meshes.json'
    WriteInJson(filepath, 'x_motion_meshes.json', x_motion_meshes)
    filepath = path + 'y_motion_meshes.json'
    WriteInJson(filepath, 'y_motion_meshes.json', y_motion_meshes)
    filepath = path + 'new_x_motion_meshes.json'
    WriteInJson(filepath, 'new_x_motion_meshes.json', new_x_motion_meshes)
    filepath = path + 'new_y_motion_meshes.json'
    WriteInJson(filepath, 'new_y_motion_meshes.json', new_y_motion_meshes)
    rel_x_motion_meshes = np.delete(new_x_motion_meshes,0,axis=2)-np.delete(new_x_motion_meshes,-1,axis=2)
    rel_y_motion_meshes = np.delete(new_y_motion_meshes,0,axis=2)-np.delete(new_y_motion_meshes,-1,axis=2)
    filepath = path + 'rel_x_motion_meshes.json'
    WriteInJson(filepath, 'rel_x_motion_meshes.json', rel_x_motion_meshes)
    filepath = path + 'rel_y_motion_meshes.json'
    WriteInJson(filepath, 'rel_y_motion_meshes.json', rel_y_motion_meshes)

#  读取存储的newpath json文件,即读取中间路径变量
def GetNewJsonPath(videoname):
    new = []
    path = './TempData/' + videoname + '/'
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
    file = os.listdir(path)
    for i in file:
        if '.json' in i and 'new' in i:
            filepath = path + i
            with open('./TempData/'+videoname+'/'+i, 'r', encoding='utf-8') as f:
                new.append(json.load(f))
                print('已读取'+ i +' 文件')
    return np.array(new)

#  读取存储的newpath json文件,即读取中间路径变量
def GetRelJsonPath(videoname):
    rel = []
    path = './TempData/' + videoname + '/'
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
    file = os.listdir(path)
    for i in file:
        if '.json' in i and 'rel' in i:
            filepath = path + i
            with open('./TempData/'+videoname+'/'+i, 'r', encoding='utf-8') as f:
                rel.append(json.load(f))
                print('已读取'+ i +' 文件')
    return np.array(rel)

#  操作3：生成稳定视频(此函数暂有问题未解决，可暂不使用)
# @measure_time
# def GenerateStabilizedVideo(cap, x_motion_meshes, y_motion_meshes, new_x_motion_meshes, new_y_motion_meshes):
#     """
#     @param: cap is the cv2.VideoCapture object that is
#             instantiated with given video
#     @param: x_motion_meshes is the motion vectors on
#             mesh vertices in x-direction
#     @param: y_motion_meshes is the motion vectors on
#             mesh vertices in y-direction
#     @param: new_x_motion_meshes is the updated motion vectors
#             on mesh vertices in x-direction to be warped with
#     @param: new_y_motion_meshes is the updated motion vectors
#             on mesh vertices in y-direction to be warped with
#     """
#     # 获取视频参数
#     frame_rate, frame_width, frame_height, frame_count=GetParameter(cap)
#     # 生成稳定视频
#     fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
#     out = cv2.VideoWriter('./stable.avi', fourcc, frame_rate, (frame_height ,frame_width))
#
#     frame_num = 0
#     print('frame_count',frame_count)
#     bar = tqdm(total=frame_count)
#     while frame_num < frame_count:
#         try:
#             print('第', frame_num, '帧')
#             # 按帧重构视频
#             ret, frame = cap.read()
#             print('frame.shape',frame.shape)
#             x_motion_mesh = x_motion_meshes[:, :, frame_num]
#             y_motion_mesh = y_motion_meshes[:, :, frame_num]
#             new_x_motion_mesh = new_x_motion_meshes[:, :, frame_num]
#             new_y_motion_mesh = new_y_motion_meshes[:, :, frame_num]
#             print(x_motion_mesh.shape)
#             # 网格扭曲
#             print('翘曲')
#             # n_f 形状 (360*640*3)
#             new_frame = mesh_warp_frame(frame, new_x_motion_mesh, new_y_motion_mesh)
#             print(new_frame.shape)
#             # cv2.imshow('new_frame', new_frame)
#             # cv2.waitKey()
#             # print('aaaaaaaaa',new_frame.shape)
#             # # new_frame = new_frame[HORIZONTAL_BORDER:-HORIZONTAL_BORDER, VERTICAL_BORDER:-VERTICAL_BORDER, :]
#             # print(new_frame.shape)
#             # # new_frame = cv2.resize(new_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
#             # # output (360,1280,3)
#             # # output = np.concatenate((frame, new_frame), axis=1)
#             # # print(output.shape)
#             # # out.write('output',output)
#             # print('new_frame.shape',new_frame.shape)
#             # out.write('new_frame',new_frame)
#             # print(out.shape)
#             cv2.imwrite('./new_frame' + str(frame_num) + '.jpg', new_frame)
#             # 绘制运动矢量
#             r = 5
#             for i in range(x_motion_mesh.shape[0]):
#                 for j in range(x_motion_mesh.shape[1]):
#                     theta = np.arctan2(y_motion_mesh[i, j], x_motion_mesh[i, j])
#                     cv2.line(frame, (j*PIXELS, i*PIXELS), (int(j*PIXELS+r*np.cos(theta)), int(i*PIXELS+r*np.sin(theta))), 1)
#             cv2.imwrite('./Results/old_motion_vectors/'+str(frame_num)+'.jpg', frame)
#             print('写入旧帧')
#             # 绘制新运动矢量
#             for i in range(new_x_motion_mesh.shape[0]):
#                 for j in range(new_x_motion_mesh.shape[1]):
#                     theta = np.arctan2(new_y_motion_mesh[i, j], new_x_motion_mesh[i, j])
#                     cv2.line(new_frame, (j*PIXELS, i*PIXELS), (int(j*PIXELS+r*np.cos(theta)), int(i*PIXELS+r*np.sin(theta))), 1)
#             cv2.imwrite('./Results/new_motion_vectors/'+str(frame_num)+'.jpg', new_frame)
#
#             frame_num += 1
#             bar.update(1)
#             print('frame_num',frame_num)
#         except:
#             print('出现错误!无法写入！')
#             break
#     bar.close()
#     cap.release()
#     out.release()


#  存帧
#  生成稳定帧
@measure_time
def SaveFrames(videoname,cap, new_x_motion_meshes, new_y_motion_meshes, rel_x_motion_meshes, rel_y_motion_meshes):
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0
    new_frame_path = './TempData/' + videoname + '/' + videoname + '_new_frame/'
    if os.path.isdir(new_frame_path):
        pass
    else:
        os.mkdir(new_frame_path)
    no_black_path = './TempData/' + videoname + '/' + videoname + '_no_black_frame/'
    if os.path.isdir(no_black_path):
        pass
    else:
        os.mkdir(no_black_path)
    contrast_path = './TempData/' + videoname + '/' + videoname + '_contrast/'
    if os.path.isdir(contrast_path):
        pass
    else:
        os.mkdir(contrast_path)

    old_frame = 0
    bar = tqdm(total=frame_count)
    while frame_num < frame_count:
        try:
            # 按帧重构视频
            ret, frame = cap.read()
            new_x_motion_mesh = new_x_motion_meshes[:, :, frame_num]
            new_y_motion_mesh = new_y_motion_meshes[:, :, frame_num]
            # 网格扭曲
            # n_f 形状 (360*640*3)
            new_frame = mesh_warp_frame(frame, new_x_motion_mesh, new_y_motion_mesh)
            cv2.imwrite(new_frame_path + str(frame_num) + '.jpg', new_frame)
            if frame_num != 0:
                rel_x_motion_mesh = rel_x_motion_meshes[:, :, frame_num]
                rel_y_motion_mesh = rel_y_motion_meshes[:, :, frame_num]
                no_black_frame = copy.deepcopy(new_frame)
                for i in range(0, new_frame.shape[0]):
                    for j in range(0, new_frame.shape[1]):
                        (b, g, r) = new_frame[i][j]
                        if [b, g, r] == [0, 0, 0]:
                            x = i - int(rel_x_motion_mesh[i % 16 - 1, j % 16 - 1])
                            y = j - int(rel_y_motion_mesh[i % 16 - 1, j % 16 - 1])
                            (d, e, f) = no_black_frame[i][j]
                            if[d, e, f] == [0, 0, 0]:
                                if x % 16 == 0 or y % 16 == 0:
                                    no_black_frame[i][j] = old_frame[-1][x-1][y-1]
                            else:
                                no_black_frame[i][j] = old_frame[-1][x][y]
                        else:
                            continue
                output = np.concatenate((frame, no_black_frame), axis=1)
                output = np.concatenate((output, new_frame), axis=1)
                cv2.imwrite(no_black_path + str(frame_num) + '.jpg', no_black_frame)
                cv2.imwrite(contrast_path + str(frame_num) + '.jpg', output)
            else:
                output = np.concatenate((frame, new_frame), axis=1)
                output = np.concatenate((output, new_frame), axis=1)
                cv2.imwrite(contrast_path + str(frame_num) + '.jpg', output)
                cv2.imwrite(no_black_path + str(frame_num) + '.jpg', new_frame)
            old_frame = np.append(old_frame,copy.deepcopy(new_frame),axis=0)
            # list_x = np.append(list_x,copy.deepcopy(new_x_motion_mesh),axis=0)
            # list_y = np.append(list_y,copy.deepcopy(new_y_motion_mesh),axis=0)
            if len(old_frame)>3:
                old_frame = np.delete(old_frame, 0, axis=0)
                # list_x = np.delete(list_x, 0, axis=0)
                # list_y = np.delete(list_y, 0, axis=0)
            frame_num += 1
            bar.update(1)
        except:
            print('出现错误!无法写入！')
            break
    bar.close()
    print('已保存稳定帧' + new_frame_path)
    print('已保存去黑边稳定帧' + no_black_path)
    print('已保存对比帧' + contrast_path)

#  生成去黑边稳定帧

# @measure_time
# def SaveNoBlackFrames(videoname,cap, rel_x_motion_meshes, rel_y_motion_meshes):
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_num = 0
#     #  生成目录
#     path_new = './TempData/' + videoname + '/' + videoname + '_no_black_frame/'
#     if os.path.isdir(path_new):
#         pass
#     else:
#         os.mkdir(path_new)
#     bar = tqdm(total=frame_count)
#     old_frame = 0
#     while frame_num < frame_count:
#         try:
#             # 按帧重构视频
#             ret, frame = cap.read()
#             rel_x_motion_mesh = rel_x_motion_meshes[:, :, frame_num]
#             rel_y_motion_mesh = rel_y_motion_meshes[:, :, frame_num]
#             # 网格扭曲
#             # n_f 形状 (360*640*3)
#             rel_frame = mesh_warp_frame(frame, rel_x_motion_mesh, rel_y_motion_mesh)
#             for i in range(0,rel_frame.shape[0]):
#                 for j in range(0,rel_frame.shape[1]):
#                     (b, g, r) = rel_frame[30][30]
#                     if [b,g,r]==[0,0,0]:
#                         x=i-int(rel_x_motion_mesh[i%16-1,j%16-1])
#                         y=j-int(rel_y_motion_mesh[i%16-1,j%16-1])
#                         rel_frame[i][j]=old_frame[x][y]
#                     else:
#                         continue
#             cv2.imwrite(path_new + str(frame_num) + '.jpg', rel_frame)
#             old_frame = rel_frame
#             frame_num += 1
#             bar.update(1)
#         except:
#             print("frame_num",frame_num,'出现错误!无法写入！')
#             break
#     bar.close()
#     print('已保存去黑边帧' + path_new)

#  帧合成稳定视频
@measure_time
def MakeStableVideo(videoname,cap):
    if cap:
        frame_rate, frame_width, frame_height, frame_count=GetParameter(cap)
    else:
        frames = os.listdir('./TempData/' + videoname + '/' + videoname + '_new_frame/')  # 读入文件夹
        im = cv2.imread('./TempData/' + videoname + '/' + videoname + '_new_frame/0.jpg')
        print(im.shape)
        frame_rate, frame_width, frame_height, frame_count = 30,im.shape[1],im.shape[0],len(frames)
    #设置生成路径
    path = './Results/' + videoname + '/'
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)

    # 生成视频参数设置
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 显示测试
    # cv2.imshow('frame', cv2.imread('./TempData/'+videoname+'_new_frame/' + '3' + '.jpg'))
    # cv2.waitKey()
    out = cv2.VideoWriter(path + videoname +'_stable.avi', fourcc, frame_rate, (frame_width, frame_height), True)
    bar = tqdm(total=frame_count)
    for i in range(0,frame_count):
        image_path = './TempData/' + videoname + '/' + videoname + '_new_frame/'+str(i)+'.jpg'
        frame = cv2.imread(image_path)
        out.write(frame)
        bar.update(1)
    bar.close()
    out.release()
    print('已保存稳定视频视频' + path + videoname + '_stable.avi')
#  帧合成去黑边稳定视频
@measure_time
def MakeNoBlackVideo(videoname,cap):
    if cap:
        frame_rate, frame_width, frame_height, frame_count=GetParameter(cap)
    else:
        frames = os.listdir('./TempData/' + videoname + '/' + videoname + '_no_black_frame/')  # 读入文件夹
        im = cv2.imread('./TempData/' + videoname + '/' + videoname + '_no_black_frame/0.jpg')
        print(im.shape)
        frame_rate, frame_width, frame_height, frame_count = 30,im.shape[1],im.shape[0],len(frames)
    #设置生成路径
    path = './Results/' + videoname + '/'
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)

    # 生成视频参数设置
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 显示测试
    # cv2.imshow('frame', cv2.imread('./TempData/'+videoname+'_new_frame/' + '3' + '.jpg'))
    # cv2.waitKey()
    out = cv2.VideoWriter(path + videoname +'_no_black.avi', fourcc, frame_rate, (frame_width, frame_height), True)
    bar = tqdm(total=frame_count)
    for i in range(0,frame_count):
        image_path = './TempData/' + videoname + '/' + videoname + '_no_black_frame/'+str(i)+'.jpg'
        frame = cv2.imread(image_path)
        out.write(frame)
        bar.update(1)
    bar.close()
    out.release()
    print('已保存去黑边稳定视频视频' + path + videoname + '_no_black.avi')

#  帧合成对比视频
@measure_time
def MakeConstrastVideo(videoname,cap):
    if cap:
        frame_rate, frame_width, frame_height, frame_count=GetParameter(cap)
    else:
        frames = os.listdir('./TempData/' + videoname + '/' + videoname + '_new_frame/')  # 读入文件夹
        im = cv2.imread('./TempData/' + videoname + '/' + videoname + '_new_frame/0.jpg')
        frame_rate, frame_width, frame_height, frame_count = 30,im.shape[1],im.shape[0],len(frames)
    frame_width = frame_width * 3
    #设置生成路径
    path = './Results/' + videoname + '/'
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
    # 生成视频参数设置
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 显示测试
    # cv2.imshow('frame', cv2.imread('./TempData/'+videoname+'_contrast/' + '3' + '.jpg'))
    # cv2.waitKey()
    out = cv2.VideoWriter(path + videoname +'_contrast.avi', fourcc, frame_rate, (frame_width, frame_height), True)
    bar = tqdm(total=frame_count)
    for i in range(0,frame_count-1):
        rer,frame = cap.read()
        new_image_path = './TempData/' + videoname + '/' + videoname + '_new_frame/'+str(i)+'.jpg'
        no_black_frame_path = './TempData/' + videoname + '/' + videoname + '_no_black_frame/'+str(i)+'.jpg'
        new_frame = cv2.imread(new_image_path)
        no_black_frame = cv2.imread(no_black_frame_path)
        temp = np.concatenate((new_frame, no_black_frame), axis=1)
        contrast_frame = np.concatenate((temp , frame), axis=1)
        out.write(contrast_frame)
        bar.update(1)
    bar.close()
    out.release()
    print('已保存对比视频'+path + videoname +'_contrast.avi')
#  帧合成50帧短视频
@measure_time
def MakeShortVideo(videoname,cap):
    frame_rate, frame_width, frame_height, frame_count=GetParameter(cap)
    path = './Results/' + videoname + '/'
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
    # 生成视频参数设置
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 显示测试
    # cv2.imshow('frame', cv2.imread('./TempData/'+videoname+'_new_frame/' + '3' + '.jpg'))
    # cv2.waitKey()
    out = cv2.VideoWriter(path + videoname +'_short.avi', fourcc, frame_rate, (frame_width, frame_height), True)
    bar = tqdm(total=frame_count)
    for i in range(0,50):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        out.write(frame)
        bar.update(1)
    bar.close()
    out.release()
    print('已保存稳定视频视频' + path + videoname + '_short.avi')
#  计划抽出单应计算转gpu进行加速
def get_H():
    pass

