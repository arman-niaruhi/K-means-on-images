import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage.morphology import binary_opening, binary_fill_holes
from skimage import measure
import math

class kmeans():

    def __init__(self) -> None:
        self.segmentationMap = None

    def getLargestCC(segmentation):
        labels = measure.label(segmentation)
        assert( labels.max() != 0 ) # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        return largestCC


    def kmeanOnAnImage(self,testimg,kkk):
        #K-MEANS algorithm
        # This is not a good initialization: Sometimes we get empty clusters!
        ny,nx,nc = testimg.shape
        c = np.random.rand(nc,kkk)

        maxIter = 15
        for i in range(0,maxIter):
            # step 1: update mask
            diff = np.zeros((ny,nx,kkk))
            for l in range(0,kkk):
                cl_3d = c[:,l][np.newaxis, np.newaxis,:]
                cl_image = np.repeat(np.repeat(cl_3d, ny, axis=0), nx, axis=1)
                diff[:,:,l] = np.sum((cl_image- testimg)**2,axis = 2)

            indi = np.argmin(diff, axis=2)


            # stept 2: update colors
            for l in range(0,kkk):
                ml = indi==l
                count = np.max([np.sum(ml),1])
                ml_3d = np.repeat(ml[:,:,np.newaxis], nc, axis=2)
                c[:,l] = 1/count*(ml_3d*testimg).sum(axis=0).sum(axis=0)

            approxImg = np.zeros((ny,nx,nc))
            for l in range(0,kkk):
                    ml = indi==l
                    approxImg = approxImg + ml[:,:,np.newaxis]* np.repeat(np.repeat(c[:,l][np.newaxis, np.newaxis,:], ny, axis=0), nx, axis=1)
                
                #return the masks
        return indi, approxImg


    def apply(self, path1, path2, k = 2):
        img=Image.open(path2)
        x, y = img.size
        ratio = x/y
        img = img.resize((math.floor(1000*ratio), 1000), Image.LANCZOS)
        beachImg=np.array(img)/255.0
        plt.imshow(beachImg)
        plt.style.use(['dark_background'])
        plt.box(False)
        plt.savefig('./././static/images/back_img1.png')

        img1=Image.open(path1)
        x, y = img1.size
        ratio = x/y
        img1 = img1.resize((math.floor(400*ratio), 400), Image.LANCZOS)
        catImg=np.array(img1)/255.0
        plt.imshow(catImg)
        plt.style.use(['dark_background'])
        plt.box(False)
        plt.savefig('./././static/images/fore_img1.png')


        catImgNormalized = catImg / np.maximum(catImg.mean(axis=2)[:,:,np.newaxis],2/255)
        catImgNormalized = catImgNormalized / np.max(catImgNormalized)
        catImgNormalized2 = catImg / np.maximum(np.sqrt(np.sum(catImg**2,axis=2))[:,:,np.newaxis],5/255)
        self.segmentationMap = catImgNormalized2[:,:,1]
        indi, approxImg = self.kmeanOnAnImage(testimg = catImgNormalized2,kkk=k)

        for i in range(k):
            plt.imshow(indi==i, cmap="binary")
            plt.style.use(['dark_background'])
            plt.box(False)
            plt.savefig(f'./././static/images/fig--{i}.png')
        return indi, catImg, beachImg


        
    def combine(self, numbers, indi, catImg, beachImg):

        # Based on outcomes of he last function combine the images
        import math
        for i in numbers:

            self.segmentationMap += (indi == (int(i)-1))

        manipulatedImg = beachImg.copy()
        segmentationMap = self.segmentationMap
        segmentationMap = binary_fill_holes(segmentationMap)
        se = np.ones((11,11), dtype=bool)
        segmentationMap = binary_opening(segmentationMap,se)
        # print(segmentationMap.shape)
        # segmentationMap=self.getLargestCC(segmentationMap[0])

        ny,nx,nc = catImg.shape
        manipulatedImg[-ny-1:-1, math.floor(nx/4):math.floor(5*nx/4),:] = catImg*~segmentationMap[:,:,np.newaxis] + (1-~segmentationMap[:,:,np.newaxis])*beachImg[-ny-1:-1, math.floor(nx/4):math.floor(5*nx/4) ,:]
        #plt.imshow(catImg*segmentationMap[:,:,np.newaxis])
        plt.imshow(manipulatedImg)
        plt.style.use(['dark_background'])
        plt.box(False)
        plt.savefig('./././static/images/comb.png')