#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//Computes a single row of the destination image by summing radius pixels
//Parameters: src: Teh src image as width*height*bpp 1d array
//            dest: pre-allocated array of size width*height*bpp to receive summed row
//            row: The current row number
//            pWidth: The width of the image * the bpp (i.e. number of bytes in a row)
//            rad: the width of the blur
//            bpp: The bits per pixel in the src image
//Returns: None
__global__ void computeRow(float* src,float* dest,int pWidth,int height,int radius,int bpp){
	int i;
    	int bradius=radius*bpp;
	int row=blockIdx.x*blockDim.x+threadIdx.x;
	if(row<height){

    		//initialize the first bpp elements so that nothing fails
    		for (i=0;i<bpp;i++)
        		dest[row*pWidth+i]=src[row*pWidth+i];
    			//start the sum up to radius*2 by only adding (nothing to subtract yet)
    		for (i=bpp;i<bradius*2*bpp;i++)
        		dest[row*pWidth+i]=src[row*pWidth+i]+dest[row*pWidth+i-bpp];
     		for (i=bradius*2+bpp;i<pWidth;i++)
        		dest[row*pWidth+i]=src[row*pWidth+i]+dest[row*pWidth+i-bpp]-src[row*pWidth+i-2*bradius-bpp];
    			//now shift everything over by radius spaces and blank out the last radius items to account for sums at the end of the kernel, instead of the middle
    		for (i=bradius;i<pWidth;i++){
        		dest[row*pWidth+i-bradius]=dest[row*pWidth+i]/(radius*2+1);
    		}	
    		//now the first and last radius values make no sense, so blank them out
    		for (i=0;i<bradius;i++){
        		dest[row*pWidth+i]=0;
        		dest[(row+1)*pWidth-1-i]=0;
    		}
	}	

}

//Computes a single column of the destination image by summing radius pixels
//Parameters: src: Teh src image as width*height*bpp 1d array
//            dest: pre-allocated array of size width*height*bpp to receive summed row
//            col: The current column number
//            pWidth: The width of the image * the bpp (i.e. number of bytes in a row)
//            height: The height of the source image
//            radius: the width of the blur
//            bpp: The bits per pixel in the src image
//Returns: None
__global__ void computeColumn(uint8_t* src,float* dest,int pWidth,int height,int radius,int bpp){
	int i;
	int col=blockIdx.x*blockDim.x+threadIdx.x;
	if(pWidth>col){
		dest[col]=src[col];
    		//start tue sum up to radius*2 by only adding
    		for (i=1;i<=radius*2;i++)
        		dest[i*pWidth+col]=src[i*pWidth+col]+dest[(i-1)*pWidth+col];
    		for (i=radius*2+1;i<height;i++)
        		dest[i*pWidth+col]=src[i*pWidth+col]+dest[(i-1)*pWidth+col]-src[(i-2*radius-1)*pWidth+col];
    		//now shift everything up by radius spaces and blank out the last radius items to account for sums at the end of the kernel, instead of the middle
    		for (i=radius;i<height;i++){
        		dest[(i-radius)*pWidth+col]=dest[i*pWidth+col]/(radius*2+1);
    		}
   		 //now the first and last radius values make no sense, so blank them out
    		for (i=0;i<radius;i++){
        		dest[i*pWidth+col]=0;
        		dest[(height-1)*pWidth-i*pWidth+col]=0;
    		}
	}

}

//Usage: Prints the usage for this program
//Parameters: name: The name of the program
//Returns: Always returns -1
int Usage(char* name){
    printf("%s: <filename> <blur radius>\n\tblur radius=pixels to average on any side of the current pixel\n",name);
    return -1;
}

int main(int argc,char** argv){
    float t1,t2;
    int radius=0;
    int i;
    int width,height,bpp,pWidth;
    char* filename;
    uint8_t *img;
    uint8_t *dImg;
    float* dest,*mid,*hDest;


    if (argc!=3)
        return Usage(argv[0]);
    filename=argv[1];
    sscanf(argv[2],"%d",&radius);
   
    img=stbi_load(filename,&width,&height,&bpp,0);

    pWidth=width*bpp;  //actual width in bytes of an image row

    cudaMalloc(&mid,sizeof(float)*pWidth*height);   
    cudaMalloc(&dest,sizeof(float)*pWidth*height);
    cudaMalloc(&dImg,sizeof(uint8_t)*pWidth*height);
    hDest=(float*)malloc(sizeof(float)*pWidth*height);
    cudaMemcpy(dImg,img,pWidth*height*sizeof(uint8_t),cudaMemcpyHostToDevice);

    t1=clock();
    int blockSize=256;
    int gridSize_Col=(pWidth+blockSize-1)/blockSize;
    computeColumn<<<gridSize_Col,blockSize>>>(dImg,mid,pWidth,height,radius,bpp);
    stbi_image_free(img);
    cudaDeviceSynchronize();
    cudaMallocManaged(&img, sizeof(uint8_t)*pWidth*height);

    int gridSize_Row=(height+blockSize-1)/blockSize;
    computeRow<<<gridSize_Row,blockSize>>>(mid,dest,pWidth,height,radius,bpp);
    cudaFree(mid);
    cudaDeviceSynchronize();
    t2=clock();
    
    cudaMemcpy(hDest,dest,sizeof(float)*pWidth*height,cudaMemcpyDeviceToHost);
    //   img=(uint8_t*)malloc(sizeof(uint8_t)*pWidth*height);

    for(i=0;i<pWidth*height;i++){
	    img[i]=(uint8_t)hDest[i];
    }
    stbi_write_png("output.png",width,height,bpp,img,bpp*width);
    cudaFree(dest);
    cudaFree(img);

    printf("Blur with radius %d complete in %f seconds\n",radius,(t2-t1)/CLOCKS_PER_SEC);
}
