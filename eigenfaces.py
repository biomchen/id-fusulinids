import os, cv2, csv
import numpy as np
from matplotlib import pyplot as plt


def main():
    path = input("Enter directory:")
    #path = 'examples'
    file_list = [x for x in os.listdir(path)]
    species_names = [x[:3] for x in file_list]
    dim = np.int(np.sqrt(len(file_list)))+1

    with open('resutls_orginal_img_data.csv','w') as csvFile:
        writer = csv.writer(csvFile)
        all_img_data = []
        for i in range(0, len(file_list)):
            img =  cv2.imread(path + file_list[i], 0) # -1, 0, 1 means
            print(img)
            # cv2.imshow does not work well
            #cv2.namedWindow('Traning Images', cv2.WINDOW_AUTOSIZE)
            #cv2.imshow('Traning Images', img)
            plt.subplot(dim,dim,i+1)
            plt.imshow(img, cmap='gray',
                       interpolation = 'bicubic')
            plt.xticks([]), plt.yticks([])
            img_data = []
            for j in range(img.shape[0]):
                temp = img[j]
                img_data = np.concatenate((img_data, temp), axis=0)
            print(img_data)
            writer.writerow(img_data)
            all_img_data.append(img_data)

    all_img_data =  np.asarray(all_img_data)
    print('The dimension of the data of all images is ', all_img_data.shape)
    plt.show()

    # normalize the images
    # Chosen std and mean
    # it can be any number that it is close to std and mean
    ustd = 80
    um = 100

    with open('results_normalized_img_data.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        norm_img = []
        for n in range(all_img_data.shape[0]):
            mean_img = all_img_data[n,].mean()
            std_img = all_img_data[n,].std()
            temp_img = (all_img_data[n,]-mean_img)*ustd/std_img+um
            temp_img = temp_img.astype(int)
            # remove all zeros in the data
            temp_img2 =  [0 if i < 0 else i for i in temp_img]
            temp_img2 = np.asarray(temp_img2)
            writer.writerow(temp_img2)
            norm_img.append(temp_img2)
            plt.subplot(dim, dim, n+1)
            plt.imshow(temp_img2.reshape(300, 300),
                       cmap='gray',
                       interpolation = 'bicubic')
            plt.xticks([]), plt.yticks([])
            #cv2.imshow(temp_img.reshape((300, 300))
            print('The data of the No.{} image is'.format(n), temp_img)

    norm_img =  np.asarray(norm_img)
    print('The dimension of the data of normalized images is ', norm_img.shape)

# Mean images
    mean_norm_img = np.mean(norm_img,
                            axis = 0,
                            dtype = np.int).reshape((300,300))
    print('The mean of the normalized image is',mean_norm_img)
    plt.show()
    # cv2.imshow codes does not work well
    #cv2.namedWindow('Mean Images', cv2.WINDOW_AUTOSIZE)
    #cv2.imshow('Mean Images', mean_norm_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # plot the mean normalized images of all professors
    plt.imshow(mean_norm_img,
               cmap='gray',
               interpolation = 'bicubic')
    plt.show()
    # calculate the eigenvectors of the normlized image dataset
    print('Calculating PCA ', end = '......')
    mean, eigenvectors = cv2.PCACompute(all_img_data,
                                        mean = None,
                                        maxComponents = 5)
    print('Done')
    eigenfaces=[]
    for eigenvector in eigenvectors:
    eigenface =  eigenvector.reshape((300,300))
    eigenfaces.append(eigenface)
    print(eigenfaces)

if __name__ == '__main__':
    main()
