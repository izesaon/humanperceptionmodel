import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import urllib
import numpy as np

import tensorflow as tf


url = ("https://i.ytimg.com/vi/SfLV8hD7zX4/maxresdefault.jpg")
    
# Open specified url and load image as a string
image_string = urllib.request.urlopen(url).read()
# print(image_string)
    
# Decode string into matrix with intensity values
print("--------------------------------------------------")
imageToOpen = tf.image.decode_jpeg(image_string, channels=3)

# with tf.Session() as sess:

    # result=sess.run(imageToOpen)
    # print (result.shape)
    # a,b,c=result[:,:,0],result[:,:,1],result[:,:,2]
    # mean=np.mean(result.reshape(-1,3),axis=0)
    # print(mean)
    # result=result.reshape(-1,3)
    # print(result)
    # result=np.subtract(result,mean)
    # print(result)
    # finalResult=result.reshape(720,1280,3)
    # min=np.amin(result)
    # print("this is min " + str(min))
    # max=np.amax(result)
    # print("this is max "+ str(max))
    # print(finalResult.shape)
    # normalization=min/(max-min)
    # print("this is normalization "+ str(normalization))
    # finalResult=(finalResult-min)/(max-min)

    # print(finalResult)

    # # print(result)
    # plt.figure()
    # plt.imshow(finalResult)
    # plt.show()


a = np.array([[[1, 2], [3, 4]],[[5, 6], [7, 8]]])
d=np.mean(a, axis=2)
dd=np.arange(9.0)


dee=10**-4
print("@@@@@@@@@@@@@@@@@@@@@@@")
print(str(dee))

# plt.figure()
# plt.imshow(a)
# plt.show()

# import urllib
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import numpy as np
# # from cifar10WithNumpy import main

# # imagesFromNumpy,labelsFromNumpy,oneHotEncoding=main()
# # imagePlace = tf.placeholder(tf.float32, shape=(50000, 3,32,32,3))
# # labelPlace=tf.placeholder(tf.float32, shape=(50000))

# # data_input_image=tf.consta
# # data_input_label=tf.convert_to_tensor(labelPlace)

# data = np.arange(1, 50000 + 1)
# data_input = tf.constant(data)

# # batch_shuffle = tf.train.shuffle_batch([data_input], enqueue_many=True, batch_size=128, capacity=100, min_after_dequeue=10,allow_smaller_final_batch=True)
# # # batch_no_shuffle = tf.train.batch([data_input], enqueue_many=True, batch_size=10, capacity=100, allow_smaller_final_batch=True)

# # with tf.Session() as sess:

# #     coord = tf.train.Coordinator()
# #     sess.run(tf.local_variables_initializer())
# #     threads = tf.train.start_queue_runners(coord=coord)
# #     sess.run(data_input)
# #     # sess.run(data_input_label, feed_dict={labelPlace: labelsFromNumpy})
# #     # for i in range(int(50000/128)):
# #     	# print(sess.run([batch_shuffle]))
# #     	# print(batch_shuffle.shape)
# #     coord.request_stop()
# #     coord.join(threads)

# # create a file-like object from the url
# # f = urllib.request.urlopen("https://i.ytimg.com/vi/SfLV8hD7zX4/maxresdefault.jpg").read()
# # print(f)

# p=np.array([[1,2],[3,4]])
# print(p)
# p=np.append(p,[[5,6]],0)
# print(p)
# p=np.append(p,[[7,8]],0)
# print(p)

# arr2=np.ones(shape=(2,2,2,2,2))
# mean_values=arr2.mean(axis=(2,3))

# print(arr2)
# print("GOOD")
# print(mean_values)


# # arr = np.array([[[1,2,3,4], [5,6,7,8]]],[[[1,2,3,4], [5,6,7,8]]])



# # image=tf.image.decode_jpeg(f,channels=3)
# # # image=tf.reshape(image,[-1,12,12,1])
# # finalImage=tf.image.central_crop(image,0.5)
# # float_image = tf.image.per_image_standardization(finalImage)
# # r,g,b=finalImage[:,:,0],finalImage[:,:,1],finalImage[:,:,2]
# # print(image)
# # print("YO")
# # print(b)

# # # image=["cat,","dog"]

# # multiply=tf.constant([3])

# # print(multiply[0])

# # a=np.matrix([1,2])

# # print(np.repeat(a,3))




# # # image1=tf.shape(split0)
# # # image2=tf.shape(split1)
# # # image3=tf.shape(split2)

# # print(finalImage.get_shape())
# # print("printing")
# # vec=tf.constant([1,2,3,4])
# # hey=tf.shape(vec)[0]
# # # print(split0)

# # with tf.Session() as sess:
# #     with tf.device("/cpu:0"):
       
# #         # np_image=sess.run(image)
# #         hey=sess.run(r)
# #         dude=sess.run(finalImage)
# #         print(len(dude))
# #         print(hey)
# #         np_image2=sess.run(r)
# #         np_image3=sess.run(g)
# #         np_image4=sess.run(b)

        
# #         print("the shape")
# #         print(len(hey))
# #         # print(np_image)

# # # plt.figure()
# # # plt.imshow(np_image3.astype(np.uint8))
# # # plt.show()

# # arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# # print (arr[:,:,0])

# # for imageIndex in range(5):
# #     print("YO")

