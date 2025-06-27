#include <stdio.h>
#include <stdlib.h>
#include <direct.h>
#include <float.h>
#include <algorithm>

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
} FullyConnectLayer;

typedef struct {
    int kernel_size;
    int stride;
} MaxPooling;
#pragma pack(pop)
// 网络结构定义
ConvolutionLayer conv1, conv2, conv3, conv4;
MaxPooling pool1, pool2, pool3;
FullyConnectLayer fc1, fc2, fc3;

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
    printf("saved to output folder %s ...\n", filename);
    return;
}

void readFloatBinary(const char *filename, float **data, int num_floats) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Failed to open file: %s\n", filename);
        return;
    }

    *data = (float *)malloc(num_floats * sizeof(float));
    if (*data == NULL) {
        printf("Memory allocation failed.\n");
        fclose(file);
        return;
    }

    fread(*data, sizeof(float), num_floats, file);
    fclose(file);
}

void initConv(ConvolutionLayer *conv, int in_channel, int out_channel, int kernel_size, int stride, int padding, int weight_index, int bias_index)
{
    // 权重和偏置的全局偏移量
    static int total_weights = 0;
    static int total_biases = 0;
    
    // 计算当前层需要的参数数量
    int num_weights = out_channel * in_channel * kernel_size * kernel_size;
    int num_biases = out_channel;
    
    // 读取权重
    static float* all_weights = NULL;
    static float* all_biases = NULL;
    
    if (weight_index == 0) {
        // 第一次调用时读取整个文件
        FILE* weight_file = fopen("../cnn_python/params/weight_conv.bin", "rb");
        if (weight_file) {
            fseek(weight_file, 0, SEEK_END);
            long file_size = ftell(weight_file);
            fseek(weight_file, 0, SEEK_SET);
            int num_floats = file_size / sizeof(float);
            all_weights = (float*)malloc(num_floats * sizeof(float));
            fread(all_weights, sizeof(float), num_floats, weight_file);
            fclose(weight_file);
        }
        
        FILE* bias_file = fopen("../cnn_python/params/bias_conv.bin", "rb");
        if (bias_file) {
            fseek(bias_file, 0, SEEK_END);
            long file_size = ftell(bias_file);
            fseek(bias_file, 0, SEEK_SET);
            int num_floats = file_size / sizeof(float);
            all_biases = (float*)malloc(num_floats * sizeof(float));
            fread(all_biases, sizeof(float), num_floats, bias_file);
            fclose(bias_file);
        }
    }
    
    // 分配当前层的内存
    conv->weight = (float*)malloc(num_weights * sizeof(float));
    conv->bias = (float*)malloc(num_biases * sizeof(float));
    
    // 复制参数
    if (all_weights && all_biases) {
        std::copy(all_weights + total_weights, all_weights + total_weights + num_weights, conv->weight);
        std::copy(all_biases + total_biases, all_biases + total_biases + num_biases, conv->bias);
    }
    
    // 更新全局偏移量
    total_weights += num_weights;
    total_biases += num_biases;
    
    // 设置其他参数
    conv->in_channel = in_channel;
    conv->out_channel = out_channel;
    conv->kernel_size = kernel_size;
    conv->stride = stride;
    conv->padding = padding;
}

void initFc(FullyConnectLayer *fc, int in_channel, int out_channel, int weight_index, int bias_index) 
{
    // 权重和偏置的全局偏移量
    static int total_weights = 0;
    static int total_biases = 0;
    
    // 计算当前层需要的参数数量
    int num_weights = out_channel * in_channel;
    int num_biases = out_channel;
    
    // 读取权重
    static float* all_weights = NULL;
    static float* all_biases = NULL;
    
    if (weight_index == 0) {
        // 第一次调用时读取整个文件
        FILE* weight_file = fopen("../cnn_python/params/weight_fc.bin", "rb");
        if (weight_file) {
            fseek(weight_file, 0, SEEK_END);
            long file_size = ftell(weight_file);
            fseek(weight_file, 0, SEEK_SET);
            int num_floats = file_size / sizeof(float);
            all_weights = (float*)malloc(num_floats * sizeof(float));
            fread(all_weights, sizeof(float), num_floats, weight_file);
            fclose(weight_file);
        }
        
        FILE* bias_file = fopen("../cnn_python/params/bias_fc.bin", "rb");
        if (bias_file) {
            fseek(bias_file, 0, SEEK_END);
            long file_size = ftell(bias_file);
            fseek(bias_file, 0, SEEK_SET);
            int num_floats = file_size / sizeof(float);
            all_biases = (float*)malloc(num_floats * sizeof(float));
            fread(all_biases, sizeof(float), num_floats, bias_file);
            fclose(bias_file);
        }
    }
    
    // 分配当前层的内存
    fc->weight = (float*)malloc(num_weights * sizeof(float));
    fc->bias = (float*)malloc(num_biases * sizeof(float));
    
    // 复制参数
    if (all_weights && all_biases) {
        std::copy(all_weights + total_weights, all_weights + total_weights + num_weights, fc->weight);
        std::copy(all_biases + total_biases, all_biases + total_biases + num_biases, fc->bias);
    }
    
    // 更新全局偏移量
    total_weights += num_weights;
    total_biases += num_biases;
    
    // 设置其他参数
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
    output->width = input->width + 2 * padding;
    output->height = input->height + 2 * padding;
    output->channel = input->channel;
    
    int output_size = output->width * output->height * output->channel;
    output->data = (float *)malloc(output_size * sizeof(float));
    if (output->data == NULL) {
        printf("Memory allocation failed for padding.\n");
        return;
    }
    
    // 初始化为0
    for (int i = 0; i < output_size; i++) {
        output->data[i] = 0.0f;
    }
    
    // 复制原始图像到中心区域
    for (int c = 0; c < input->channel; c++) {
        for (int i = 0; i < input->height; i++) {
            for (int j = 0; j < input->width; j++) {
                int in_idx = c * input->width * input->height + i * input->width + j;
                int out_idx = c * output->width * output->height + 
                             (i + padding) * output->width + (j + padding);
                output->data[out_idx] = input->data[in_idx];
            }
        }
    }
}
 
void image2vector(Image *input, Vector *output)
{
    output->channel = input->channel * input->width * input->height;
    output->data = (float *)malloc(output->channel * sizeof(float));
    if (output->data == NULL) {
        printf("Memory allocation failed for vector.\n");
        return;
    }
    
    // 按通道优先顺序展平
    for (int i = 0; i < output->channel; i++) {
        output->data[i] = input->data[i];
    }
}

void convolve(Image *input, Image *output, const ConvolutionLayer conv)
{      
    // 应用padding
    Image *padded = (Image *)malloc(sizeof(Image));
    pad(input, padded, conv.padding);
    
    output->channel = conv.out_channel;
    output->height = (padded->height - conv.kernel_size) / conv.stride + 1;
    output->width = (padded->width - conv.kernel_size) / conv.stride + 1;
    
    int output_size = output->channel * output->height * output->width;
    output->data = (float *)malloc(output_size * sizeof(float));
    if (output->data == NULL) {
        printf("Memory allocation failed for convolution output.\n");
        free(padded->data);
        free(padded);
        return;
    }
    
    // 执行卷积
    for (int oc = 0; oc < conv.out_channel; oc++) {
        for (int oh = 0; oh < output->height; oh++) {
            for (int ow = 0; ow < output->width; ow++) {
                int h_start = oh * conv.stride;
                int w_start = ow * conv.stride;
                float sum = 0.0f;
                
                for (int ic = 0; ic < conv.in_channel; ic++) {
                    for (int kh = 0; kh < conv.kernel_size; kh++) {
                        for (int kw = 0; kw < conv.kernel_size; kw++) {
                            int h_idx = h_start + kh;
                            int w_idx_img = w_start + kw;  // 重命名这个变量
                            
                            // 输入索引
                            int in_idx = ic * padded->width * padded->height + 
                                         h_idx * padded->width + w_idx_img;
                                        
                            // 权重索引 - 使用不同的变量名
                            int w_idx_weights = oc * conv.in_channel * conv.kernel_size * conv.kernel_size +
                                               ic * conv.kernel_size * conv.kernel_size +
                                               kh * conv.kernel_size + kw;
                            
                            sum += padded->data[in_idx] * conv.weight[w_idx_weights];
                        }
                    }
                }
                
                // 添加偏置
                int out_idx = oc * output->height * output->width + 
                              oh * output->width + ow;
                output->data[out_idx] = sum + conv.bias[oc];
            }
        }
    }
    
    free(padded->data);
    free(padded);
}


void max_pooling(Image *input, Image *output, const MaxPooling pool)
{
    output->channel = input->channel;
    output->height = (input->height - pool.kernel_size) / pool.stride + 1;
    output->width = (input->width - pool.kernel_size) / pool.stride + 1;
    
    int output_size = output->channel * output->height * output->width;
    output->data = (float *)malloc(output_size * sizeof(float));
    if (output->data == NULL) {
        printf("Memory allocation failed for pooling output.\n");
        return;
    }
    
    // 执行最大池化
    for (int c = 0; c < input->channel; c++) {
        for (int oh = 0; oh < output->height; oh++) {
            for (int ow = 0; ow < output->width; ow++) {
                int h_start = oh * pool.stride;
                int w_start = ow * pool.stride;
                float max_val = -FLT_MAX;
                
                for (int kh = 0; kh < pool.kernel_size; kh++) {
                    for (int kw = 0; kw < pool.kernel_size; kw++) {
                        int h_idx = h_start + kh;
                        int w_idx = w_start + kw;
                        
                        if (h_idx < input->height && w_idx < input->width) {
                            int in_idx = c * input->width * input->height + 
                                         h_idx * input->width + w_idx;
                            if (input->data[in_idx] > max_val) {
                                max_val = input->data[in_idx];
                            }
                        }
                    }
                }
                
                int out_idx = c * output->height * output->width + 
                              oh * output->width + ow;
                output->data[out_idx] = max_val;
            }
        }
    }
}

void reluImage(Image* input, Image* output) {
    output->channel = input->channel;
    output->width = input->width;
    output->height = input->height;
    
    int size = input->channel * input->width * input->height;
    output->data = (float *)malloc(size * sizeof(float));
    if (output->data == NULL) {
        printf("Memory allocation failed for ReLU output.\n");
        return;
    }
    
    for (int i = 0; i < size; i++) {
        output->data[i] = input->data[i] > 0 ? input->data[i] : 0.0f;
    }
}

void reluVector(Vector *input, Vector *output)
{   
    output->channel = input->channel;
    output->data = (float *)malloc(output->channel * sizeof(float));
    if (output->data == NULL) {
        printf("Memory allocation failed for vector ReLU.\n");
        return;
    }
    
    for (int i = 0; i < input->channel; i++) {
        output->data[i] = input->data[i] > 0 ? input->data[i] : 0.0f;
    }
}

void fullyconnect(Vector *input, Vector *output, const FullyConnectLayer fc)
{
    output->channel = fc.out_channel;
    output->data = (float *)malloc(output->channel * sizeof(float));
    if (output->data == NULL) {
        printf("Memory allocation failed for FC output.\n");
        return;
    }
    
    // 初始化为偏置
    for (int i = 0; i < output->channel; i++) {
        output->data[i] = fc.bias[i];
    }
    
    // 矩阵乘法
    for (int i = 0; i < output->channel; i++) {
        for (int j = 0; j < fc.in_channel; j++) {
            output->data[i] += input->data[j] * fc.weight[i * fc.in_channel + j];
        }
    }
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

int main()
{
    if (_mkdir("output") == 0) {
        printf("create folder output\n");
    }
	
    Image *bmp_image = (Image*)malloc(sizeof(Image));
    readBMP("./image.bmp", &(bmp_image->data), &(bmp_image->channel), &(bmp_image->width), &(bmp_image->height));   
    savetxt(bmp_image->data, 40, 40, 3, "./output/image.txt");    

    // 初始化模型参数
    initConv(&conv1, 3, 32, 3, 1, 1, 0, 0);
    initConv(&conv2, 32, 32, 3, 1, 1, 1, 1);
    initConv(&conv3, 32, 64, 3, 1, 1, 2, 2);
    initConv(&conv4, 64, 128, 3, 1, 1, 3, 3);
    
    initPool(&pool1, 2, 2);
    initPool(&pool2, 2, 2);
    initPool(&pool3, 2, 2);
    
    initFc(&fc1, 128 * 5 * 5, 256, 0, 0);
    initFc(&fc2, 256, 128, 1, 1);
    initFc(&fc3, 128, 7, 2, 2);

    // 前向传播
    Image* conv1_out = (Image*)malloc(sizeof(Image));
    convolve(bmp_image, conv1_out, conv1);
    
    Image* relu1_out = (Image*)malloc(sizeof(Image));
    reluImage(conv1_out, relu1_out);
    
    Image* conv2_out = (Image*)malloc(sizeof(Image));
    convolve(relu1_out, conv2_out, conv2);
    
    Image* relu2_out = (Image*)malloc(sizeof(Image));
    reluImage(conv2_out, relu2_out);
    
    Image* pool1_out = (Image*)malloc(sizeof(Image));
    max_pooling(relu2_out, pool1_out, pool1);
    
    Image* conv3_out = (Image*)malloc(sizeof(Image));
    convolve(pool1_out, conv3_out, conv3);
    
    Image* relu3_out = (Image*)malloc(sizeof(Image));
    reluImage(conv3_out, relu3_out);
    
    Image* pool2_out = (Image*)malloc(sizeof(Image));
    max_pooling(relu3_out, pool2_out, pool2);
    
    Image* conv4_out = (Image*)malloc(sizeof(Image));
    convolve(pool2_out, conv4_out, conv4);
    
    Image* relu4_out = (Image*)malloc(sizeof(Image));
    reluImage(conv4_out, relu4_out);
    
    Image* pool3_out = (Image*)malloc(sizeof(Image));
    max_pooling(relu4_out, pool3_out, pool3);
    
    Image* relu5_out = (Image*)malloc(sizeof(Image));
    reluImage(pool3_out, relu5_out);
    
    Vector* fc_input = (Vector*)malloc(sizeof(Vector));
    image2vector(relu5_out, fc_input);
    
    Vector* fc1_out = (Vector*)malloc(sizeof(Vector));
    fullyconnect(fc_input, fc1_out, fc1);
    
    Vector* relu6_out = (Vector*)malloc(sizeof(Vector));
    reluVector(fc1_out, relu6_out);
    
    Vector* fc2_out = (Vector*)malloc(sizeof(Vector));
    fullyconnect(relu6_out, fc2_out, fc2);
    
    Vector* relu7_out = (Vector*)malloc(sizeof(Vector));
    reluVector(fc2_out, relu7_out);
    
    Vector* fc3_out = (Vector*)malloc(sizeof(Vector));
    fullyconnect(relu7_out, fc3_out, fc3);
    
    // 保存结果
    savetxt(relu5_out->data, relu5_out->width, relu5_out->height, relu5_out->channel, "./output/output_conv.txt");
    savetxt(fc3_out->data, fc3_out->channel, 1, 1, "./output/output_fc.txt");

    // 释放内存
    freeImage(bmp_image);
    
    freeImage(conv1_out);
    freeImage(relu1_out);
    freeImage(conv2_out);
    freeImage(relu2_out);
    freeImage(pool1_out);
    freeImage(conv3_out);
    freeImage(relu3_out);
    freeImage(pool2_out);
    freeImage(conv4_out);
    freeImage(relu4_out);
    freeImage(pool3_out);
    freeImage(relu5_out);
    
    freeVector(fc_input);
    freeVector(fc1_out);
    freeVector(relu6_out);
    freeVector(fc2_out);
    freeVector(relu7_out);
    freeVector(fc3_out);
    return 0;
}
