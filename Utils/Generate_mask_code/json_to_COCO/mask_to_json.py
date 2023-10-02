from PIL import Image
import numpy as np
import cv2
import json
import os

color_dict = {  
        76: 'suction',
        150: 'kerrisons',  # 新的类别2和8
        29: 'pituitary_rongeurs',
        226: 'retractable_knife',
        179: 'freer_elevator',
        105: 'spatula_dissector',
        38: 'dural_scissors',
        15: 'stealth_pointer',
        113: 'surgiflo',
        90: 'cup_forceps',
        53: 'ring_curette',
        192: 'cottle',
        128: 'drill',
        165: 'blakesley',
        213: 'bipolar_forceps',
        183: 'doppler',
    }
imgs = os.listdir('png')
for img in imgs:
    name = img[:-4]
    img = Image.open('png/'+img).convert('L')
    sizes = img.size
    img = np.array(img)
    # img = img /255
    # img = np.trunc(img)
    list1 = np.unique(img)
    values = list1.tolist()
    print('++++++++++',values)
    # values = [0, 105, 226]
    s = 0
    # cla_dict ={1:}
    # json_str = json.dumps('', indent=4)
    # with open('seg_json_val/'+name.split('_mask')[0]+'.json', 'w') as json_file:
    #     json_file.write('')
    pix_list = []
    for i in values:
        if i == 0:
            continue
        img = Image.open('png/'+name+'.png').convert('L')
        img = np.array(img)
        # img = img /255
        # img = np.trunc(img)
        # img = Image.fromarray(img.astype('uint8'))
        # img = np.array(img)
        s +=1
        img[img==i]=255
        # image = Image.fromarray(np.uint8(img))
        new = image = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
        list1 = np.unique(image) 
        print(list1)
        
        ret, binary = cv2.threshold(image,254,255,cv2.THRESH_BINARY) 
        # cv2.imshow("binary", binary) 
        # cv2.waitKey(0)  
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        # print((contours[0].tolist()))
        # cla_list = []
        
        # print(contours)
        for sinindex in range(len( contours)):
            in_dict ={}
            sinlist = contours[sinindex].tolist()
            # print(sinlist)
            if len(sinlist)>=100:

                a = []
                for dd in range(0,len(sinlist)-2,2) :
                    a.extend(sinlist[dd])
                # if len(a)>10:

                in_dict['label'] = color_dict[i]
                # print(str(i))
                in_dict['points'] = a
                in_dict['group_id'] = None
                in_dict['shape_type'] = "polygon"
                in_dict['flags'] = {}
                # cla_list.append(cla_dict)
                pix_list.append(in_dict)

        
        # print(len(contours))
        # cv2.drawContours(new,contours,-1,(0,255,0),3)  
        
        # cv2.imshow("img", new)  
        # cv2.waitKey(0)  
    cla_dict={}
    if len(pix_list)!=0:
    #   "version": "5.0.1",
    #   "flags": {},

        cla_dict['version'] = "5.0.1"
        cla_dict['flags'] = {}
        cla_dict['shapes'] = pix_list
        cla_dict['imagePath'] = name.split('_mask')[0]+'.jpg'
        cla_dict['imageData'] = None
        cla_dict['imageHeight'] = sizes[1]
        cla_dict['imageWidth'] = sizes[0]
        json_str = json.dumps(cla_dict,indent=4)
        with open('seg_json/'+name.split('_mask')[0]+'.json', 'a') as json_file:
            json_file.write(json_str)
