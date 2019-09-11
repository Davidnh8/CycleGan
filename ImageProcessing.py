import numpy as np

class ImageProcessing():
    #def __init__():
    
    #def transform_into_square(img,dim):
        
        
    def adjust_grey_color_distribution(self, source_img, target_RGB):
        return None
    
    def adjust_RGB_color_distribution(self, source_img, target_RGB, mode='median'):
        
        #check source_img
        assert len(source_img.shape)
        
        # check target_RGB
        assert len(target_RGB)==3
        for i in range(3):
            assert target_RGB[i]>=0
            assert target_RGB[i]<=255
        
        # check mode
        if mode not in ['median', 'mean']:
            raise ValueError("the mode must be median or mean")
        
        
        if mode=='median':
            source_img_R_center= np.median(source_img[:,:,0])
            source_img_G_center= np.median(source_img[:,:,1])
            source_img_B_center= np.median(source_img[:,:,2])
        elif mode=='mean':
            source_img_R_center= np.mean(source_img[:,:,0])
            source_img_G_center= np.mean(source_img[:,:,1])
            source_img_B_center= np.mean(source_img[:,:,2])
        else:
            raise ValueError("This should not occur. Check Code")
        
        R_adjust_ratio=target_RGB[0]/source_img_R_center
        G_adjust_ratio=target_RGB[1]/source_img_G_center
        B_adjust_ratio=target_RGB[2]/source_img_B_center
        
        new_img=np.empty_like(source_img)
        new_img[:,:,0]=source_img[:,:,0]*R_adjust_ratio
        new_img[:,:,1]=source_img[:,:,1]*G_adjust_ratio
        new_img[:,:,2]=source_img[:,:,2]*B_adjust_ratio
        
        return new_img
        