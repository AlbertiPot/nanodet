# target: 剔除非标准结尾的jpeg图片
# abstract: wider pedestrian detection有一张jped图片的最后二进制码不是jpeg格式标准的结尾0xff,0xd9,p.s. 标准开头是0xff，0xd8
#           会报Premature end of JPEG file警告

import os
import glob

if __name__ == '__main__':
    imgFolder = "/home/gbc/workspace/nanodet/data/images/sur_train"
    imgPathlist = glob.glob(os.path.join(imgFolder,"sur*.jpg"))                   # 3个通配符: "*"匹配0个或多个字符；"?"匹配单个字符；"[]"匹配指定范围内的字符，如：[0-9]匹配数字
    # assert len(imgPathlist) == 1, 'Wrong number of sur images'
    
    corrupImgdict = {}
    for index, path in enumerate(imgPathlist):
        with open(path, 'rb') as f:
            binaryImg = f.read()
            check_chars = binaryImg[-2:]
        if check_chars != b'\xff\xd9':
            corrupImgdict[index] = path.replace(imgFolder+'/','')
    
    print(corrupImgdict)                                                          # 尝试添加xd9恢复图失败
    # with open(os.path.join(imgFolder,corrupImgdict[0]), 'ab') as f:
    #     f.write(b'\xd9') 
    

            


            
    