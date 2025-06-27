#include <stdio.h>
#include <stdlib.h>
#include <direct.h>

#pragma pack(push, 1) 
typedef struct {
    unsigned short bfType;          
    unsigned int bfSize;            
    unsigned short bfReserved1;     
    unsigned short bfReserved2;     
    unsigned int bfOffBits;         
} BITMAPFILEHEADER;

typedef struct {
    unsigned int biSize;            
    int biWidth;                    
    int biHeight;                  
    unsigned short biPlanes;       
    unsigned short biBitCount;      
    unsigned int biCompression;    
    unsigned int biSizeImage;       
    int biXPelsPerMeter;            
    int biYPelsPerMeter;         
    unsigned int biClrUsed;        
    unsigned int biClrImportant;   
} BITMAPINFOHEADER;

typedef struct {
    int width;
    int height;
    int channel;   
    float* data;    
} Image;

typedef struct {
    int channel;   
    float* data;    
} Vector;

typedef struct {
    int in_channel; 
    int out_channel;     
    int kernel_size;   
    int stride;        
    int padding;       
    float *weight;
    float *bias;
} ConvolutionLayer;

typedef struct {
    int in_channel; 
    int out_channel;  
    float *weight;
    float *bias;
} FullyConnetLayer;

typedef struct {
    int kernel_size;   
    int stride;    
} MaxPooling;
#pragma pack(pop)

void readBMP(const char *filename, float **p_data, int *p_channel, int *p_width, int *p_height) 
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Failed to open BMP file.\n");
        return;
    }

    BITMAPFILEHEADER fileHeader;
    BITMAPINFOHEADER infoHeader;

    fread(&fileHeader, sizeof(BITMAPFILEHEADER), 1, file);
    fread(&infoHeader, sizeof(BITMAPINFOHEADER), 1, file);

    if (fileHeader.bfType != 0x4D42) {
        printf("Not a valid BMP file.\n");
        fclose(file);
        return;
    }

    int width = infoHeader.biWidth;
    int height = infoHeader.biHeight;

    *p_width = width;
    *p_height = height;
    *p_channel = 3;

    fseek(file, fileHeader.bfOffBits, SEEK_SET);

    int rowSize = (width * 3 + 3) & ~3; 
    unsigned char *imageData = (unsigned char *)malloc(rowSize * height);
    if (imageData == NULL) {
        printf("Memory allocation failed.\n");
        fclose(file);
        return;
    }

    fread(imageData, 1, rowSize * height, file);
    fclose(file);    

    *p_data = (float *)malloc(width * height * 3 * sizeof(float));
    if (*p_data == NULL) {
        printf("Memory allocation failed for RGB channels.\n");
        free(imageData);        
        return;
    }
    
    int pixelIndex = 0;
    float *pRGB_R;
    float *pRGB_G;
    float *pRGB_B;

    for (int y = 0; y < height; y++) {
        pRGB_R = *p_data + width * height - width * (y + 1);
        pRGB_G = *p_data + width * height * 2 - width * (y + 1);
        pRGB_B = *p_data + width * height * 3 - width * (y + 1);
        for (int x = 0; x < width; x++) {    
            *pRGB_B = (float(imageData[pixelIndex]) / 255. - 0.5) / 0.5;     
            *pRGB_G = (float(imageData[pixelIndex + 1]) / 255. - 0.5) / 0.5; 
            *pRGB_R = (float(imageData[pixelIndex + 2]) / 255. - 0.5) / 0.5; 
            pRGB_B++;
            pRGB_G++;
            pRGB_R++;    
            pixelIndex = pixelIndex + 3;        
        }
    }     

    free(imageData);    
}

void savetxt(float *p_data, int width, int height, int channel, const char *filename)
{
    FILE *outFile = fopen(filename, "w");
    if (outFile == NULL) {
        printf("Failed to create output file.\n");        
        return;
    }

    float *RGB = p_data;
    for (int i = 0; i < width * height * channel; i++) {
        fprintf(outFile, "%.7f ", RGB[i]);
        if ((i + 1) % width == 0) fprintf(outFile, "\n");
    }    

    fclose(outFile);
    printf("saved to output folder ...\n");
    return;
}

void readFloatBinary(const char *filename, float **data) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Failed to open file: %s\n", filename);
        return;
    }

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    rewind(file);

    if (fileSize % sizeof(float) != 0) {
        printf("File size is not a multiple of float size. Corrupted file?\n");
        fclose(file);
        return;
    }

    size_t numFloats = fileSize / sizeof(float);

    *data = (float *)malloc(fileSize);
    if (*data == NULL) {
        printf("Memory allocation failed.\n");
        fclose(file);  
        return;
    }

    size_t readCount = fread(*data, sizeof(float), numFloats, file);
    if (readCount != numFloats) {
        printf("Failed to read the expected number of floats.\n");
        free(data);
        fclose(file);
        return;
    }

    fclose(file);
}

void initConv(ConvolutionLayer *conv, int in_channel, int out_channel, int kernel_size, int stride, int padding)   
{        
    readFloatBinary("./params/weight_conv.bin", &(conv->weight));
    readFloatBinary("./params/bias_conv.bin", &(conv->bias));

    conv->in_channel = in_channel;
    conv->out_channel = out_channel;
    conv->kernel_size = kernel_size;
    conv->stride = stride;
    conv->padding = padding;
}

void initFc(FullyConnetLayer *fc, int in_channel, int out_channel)   
{        
    readFloatBinary("./params/weight_fc.bin", &(fc->weight));
    readFloatBinary("./params/bias_fc.bin", &(fc->bias));

    fc->in_channel = in_channel;
    fc->out_channel = out_channel;
}

void initPool(MaxPooling *pool, int kernel_size, int stride)   
{
    pool->kernel_size = kernel_size;
    pool->stride = stride;
}

void center_crop(Image *input, Image *output, int crop_size)
{
    output->width = crop_size;
    output->height = crop_size;
    output->channel = input->channel;

    int s_width  = (input->width - crop_size) / 2;
    int s_height = (input->height - crop_size) / 2;

    output->data = (float *)malloc(crop_size * crop_size * 3 * sizeof(float));
    if (output->data == NULL) {
        printf("Memory allocation failed.\n");
        return;
    }

    float *p_input  = input->data;
    float *p_output = output->data;
    for (int c=0; c<3; c++)
    {
        for (int i=s_height; i<input->height-s_height; i++)
        {
            for (int j=s_width; j<input->width-s_width; j++)
            {
                *p_output++ = *(p_input + c * input->width * input->height + i * input->width + j);
            }
        }
    }
    return;
}

void pad(Image *input, Image *output, int padding)
{    
    return;    
}
 
void image2vector(Image *input, Vector *output)
{
    return; 
}

void convolve(Image *input, Image *output, const ConvolutionLayer conv)
{      
    return;
}


void max_pooling(Image *input, Image *output, const MaxPooling pool)
{
    return;
}

void reluImage(Image *input, Image *output)
{  
    return;
}

void reluVector(Vector *input, Vector *output)
{   
    return;
}

void fullyconnect(Vector *input, Vector *output, const FullyConnetLayer fc)
{
    return;
}

void freeImage(Image* img) {
    if (img != NULL) {
        if (img->data != NULL) {
            free(img->data);  
            img->data = NULL; 
        }
        free(img);           
        img = NULL;          
    }
}

void freeVector(Vector* vec) {
    if (vec != NULL) {
        if (vec->data != NULL) {
            free(vec->data);  
            vec->data = NULL; 
        }
        free(vec);            
        vec = NULL;           
    }
}

void freeResources(
    Image* bmp_image,
    Image* conv_output, Image* pooling_output, Image* relu_output,
    Vector* fc_input, Vector* fc_output
) 
{
    freeImage(bmp_image);

    freeImage(conv_output);
    freeImage(pooling_output);
    freeImage(relu_output);

    freeVector(fc_input);
    freeVector(fc_output);
}

int main()
{
    if (_mkdir("output") == 0) {
        printf("create folder output\n");
    }
	
    Image *bmp_image = (Image*)malloc(sizeof(Image));
    readBMP("./image.bmp", &(bmp_image->data), &(bmp_image->channel), &(bmp_image->width), &(bmp_image->height));   
    savetxt(bmp_image->data, 40, 40, 3, "./output/image.txt");    
	
//    ConvolutionLayer conv;
//    FullyConnetLayer fc;
//    MaxPooling pool;
//	
//    initConv(&conv, 3, 8, 3, 1, 1);
//    initFc(&fc, 200, 7); 
//    initPool(&pool, 8, 8);    
//	
//    Image *conv_output = (Image*)malloc(sizeof(Image));
//    Image *pooling_output = (Image*)malloc(sizeof(Image));
//    Image *relu_output = (Image*)malloc(sizeof(Image));
//    convolve(bmp_image, conv_output, conv);
//    max_pooling(conv_output, pooling_output, pool);
//    reluImage(pooling_output, relu_output);   
//
//    Vector *fc_input  = (Vector*)malloc(sizeof(Vector));
//    Vector *fc_output = (Vector*)malloc(sizeof(Vector));  
//    image2vector(relu_output, fc_input);   
//    fullyconnect(fc_input, fc_output, fc);   
//	
//    savetxt(relu_output->data, relu_output->width, relu_output->height, relu_output->channel, "./output/output_conv.txt");
//    savetxt(fc_output->data, fc_output->channel, 1, 1, "./output/output_fc.txt");
//
//    freeResources(bmp_image, conv_output, pooling_output, relu_output,
//                  fc_input, fc_output);
    return 0;
}


