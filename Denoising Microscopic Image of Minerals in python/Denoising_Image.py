from skimage import io, img_as_float
from scipy import ndimage as nd
from matplotlib  import pyplot as plt
import numpy as np
img=img_as_float(io.imread("Try.jpg"))

gaussian_img=nd.gaussian_filter(img, sigma=3)
plt.imsave("Guassian.jpg", gaussian_img)

median_img=nd.median_filter(img, size=3)
plt.imsave("median.jpg", median_img)

from skimage.restoration import denoise_nl_means, estimate_sigma
sigma_est=np.mean(estimate_sigma(img, channel_axis=-1))
nlm=denoise_nl_means(img, h=1.15*sigma_est, fast_mode=False, patch_distance=3)

plt.imsave("nlm.jpg", nlm)

##multichannel=True 