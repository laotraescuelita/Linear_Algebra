import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.image import imread

def svd_1( img ):	
	img_gray = np.mean( img , -1)

	plt.imshow( img )
	plt.title( "Imagen original ")
	plt.show()

	plt.imshow( img_gray )
	plt.title( "Imagen en escala de grises ")
	plt.show()


	#Metodo para comrprimti la imagen
	U,S,V = np.linalg.svd( img_gray, full_matrices = False )
	S_D = np.diag( S )

	#vamos a mostrar lsa iamgenes después de la transformaión.
	j = 0
	fig, axes = plt.subplots(1, 3, figsize=(10, 4))

	for i in range( 5, 20, 100):
		img_aprox = U[:,:i] @ S_D[0:i,:i] @ V[:i,:]	
		axes[j].imshow( img_aprox, cmap = "gray")
		j+=1

	plt.show()

img = imread( "img.jpg")
svd_1( img )

#LA segunda forma es crear objetos sinteticos

def svd_2(A):
	n = len( A )
	plt.imshow( A, cmap="gray" )
	plt.title( "Imagen original ")
	plt.show()

	U,S,V = np.linalg.svd( A )

	imgs = []
	for i in range(n):
	    imgs.append( S[i]*np.outer(U[:,i],V[i]) )

	combined_imgs = []
	for i in range(n):
	    img = sum(imgs[:i+1])
	    combined_imgs.append(img)
	    
	fig, axes = plt.subplots(figsize = (10,4), nrows = 1, ncols = n, sharex=True, sharey=True)
	for num, ax in zip(range(n), axes):
	    ax.imshow(imgs[num], cmap='gray', vmin=0, vmax=1)
	    ax.set_title(np.round(S[num],2), fontsize=10)
	plt.show()

	fig, axes = plt.subplots(figsize = (10,4), nrows = 1, ncols = n, sharex=True, sharey=True)
	for num, ax in zip(range(n), axes):
	    ax.imshow(combined_imgs[num], cmap='gray', vmin=0, vmax=1)
	plt.show()

	return U,S,V


A = np.array([[0,1,1,0,1,1,0],
              [1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1],
              [0,1,1,1,1,1,0],
              [0,0,1,1,1,0,0],
              [0,0,0,1,0,0,0],
             ])

U,S,V = svd_2(A)


