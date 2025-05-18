#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 多維索引轉換為一維索引
int get_5d_index(int b, int c, int d, int h, int w,
                int batch_size, int channels, int depth, int height, int width) {
    return b * (channels * depth * height * width) +
           c * (depth * height * width) +
           d * (height * width) +
           h * width +
           w;
}

// 3D核心索引轉換
int get_3d_kernel_index(int d, int h, int w, int kernel_d, int kernel_h, int kernel_w) {
    return d * (kernel_h * kernel_w) + h * kernel_w + w;
}

// 量化參數結構
typedef struct {
    float input_scale;
    int input_zero_point;
    float weight_scale;
    int weight_zero_point;
    float output_scale;
    int output_zero_point;
} QuantParams;

// 3D捲積實作
void conv3d_1d_array(
    const unsigned char* input,
    const char* weights,
    const int* bias,
    unsigned char* output,
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int out_depth, int out_height, int out_width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    const QuantParams* quant_params) {
    
    // 遍歷輸出體積的每個元素
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int od = 0; od < out_depth; od++) {
                for (int oh = 0; oh < out_height; oh++) {
                    for (int ow = 0; ow < out_width; ow++) {
                        // 計算輸出索引
                        int output_idx = get_5d_index(b, oc, od, oh, ow, 
                                         batch_size, out_channels, out_depth, out_height, out_width);
                        
                        // 累積值
                        int acc = bias[oc];
                        
                        // 對每個輸入通道和卷積核位置進行計算
                        for (int ic = 0; ic < in_channels; ic++) {
                            for (int kd = 0; kd < kernel_d; kd++) {
                                for (int kh = 0; kh < kernel_h; kh++) {
                                    for (int kw = 0; kw < kernel_w; kw++) {
                                        // 計算對應的輸入位置
                                        int id = od * stride_d + kd - padding_d;
                                        int ih = oh * stride_h + kh - padding_h;
                                        int iw = ow * stride_w + kw - padding_w;
                                        
                                        // 檢查邊界
                                        if (id >= 0 && id < in_depth &&
                                            ih >= 0 && ih < in_height &&
                                            iw >= 0 && iw < in_width) {
                                            
                                            // 計算輸入和權重索引
                                            int input_idx = get_5d_index(b, ic, id, ih, iw,
                                                            batch_size, in_channels, in_depth, in_height, in_width);
                                            int weight_idx = get_3d_kernel_index(kd, kh, kw, kernel_d, kernel_h, kernel_w) +
                                                            oc * in_channels * kernel_d * kernel_h * kernel_w * ic;
                                            
                                            // 反量化, 計算
                                            int input_val = input[input_idx] - quant_params->input_zero_point;
                                            int weight_val = weights[weight_idx] - quant_params->weight_zero_point;
                                            
                                            // 累積乘積
                                            acc += input_val * weight_val;
                                        }
                                    }
                                }
                            }
                        }
                        
                        // 重新量化結果
                        float rescale = (quant_params->input_scale * quant_params->weight_scale) / quant_params->output_scale;
                        int out_val = (int)((float)acc * rescale) + quant_params->output_zero_point;
                        
                        // 確保結果在有效範圍內
                        if (out_val < 0) out_val = 0;
                        if (out_val > 255) out_val = 255;
                        
                        output[output_idx] = (unsigned char)out_val;
                    }
                }
            }
        }
    }
}

void conv3d_depthwise_1d_array(
    const unsigned char* input,
    const char* weights,
    const int* bias,
    unsigned char* output,
    int batch_size, int channels, int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int depth_multiplier,  // 每個輸入通道的輸出通道數量
    const QuantParams* quant_params) {
    
    // 注意: Depthwise捲積總輸出通道數 = 輸入通道數 * depth_multiplier
    int out_channels = channels * depth_multiplier;
    
    // 遍歷輸出體積的每個元素
    for (int b = 0; b < batch_size; b++) {
        // 處理每個輸入通道
        for (int ic = 0; ic < channels; ic++) {
            // 每個輸入通道對應depth_multiplier個輸出通道
            for (int m = 0; m < depth_multiplier; m++) {
                // 計算當前輸出通道索引
                int oc = ic * depth_multiplier + m;
                
                for (int od = 0; od < out_depth; od++) {
                    for (int oh = 0; oh < out_height; oh++) {
                        for (int ow = 0; ow < out_width; ow++) {
                            // 計算輸出索引
                            int output_idx = get_5d_index(b, oc, od, oh, ow, 
                                             batch_size, out_channels, out_depth, out_height, out_width);
                            
                            // 累積值 (應用偏置)
                            int acc = bias[oc];
                            
                            // 應用卷積核 - 注意這裡只考慮單一輸入通道
                            for (int kd = 0; kd < kernel_d; kd++) {
                                for (int kh = 0; kh < kernel_h; kh++) {
                                    for (int kw = 0; kw < kernel_w; kw++) {
                                        // 計算對應的輸入位置
                                        int id = od * stride_d + kd - padding_d;
                                        int ih = oh * stride_h + kh - padding_h;
                                        int iw = ow * stride_w + kw - padding_w;
                                        
                                        // 檢查邊界
                                        if (id >= 0 && id < in_depth &&
                                            ih >= 0 && ih < in_height &&
                                            iw >= 0 && iw < in_width) {
                                            
                                            // 計算輸入索引 - 只考慮當前通道
                                            int input_idx = get_5d_index(b, ic, id, ih, iw,
                                                            batch_size, channels, in_depth, in_height, in_width);
                                            
                                            // 計算權重索引 - 注意權重索引的計算方式
                                            // 在depthwise中，每個輸入通道有自己的kernel集合
                                            int weight_idx = get_3d_kernel_index(kd, kh, kw, kernel_d, kernel_h, kernel_w) +
                                                            oc * kernel_d * kernel_h * kernel_w;
                                            
                                            // 反量化, 計算
                                            int input_val = input[input_idx] - quant_params->input_zero_point;
                                            int weight_val = weights[weight_idx] - quant_params->weight_zero_point;
                                            
                                            // 累積乘積
                                            acc += input_val * weight_val;
                                        }
                                    }
                                }
                            }
                            
                            // 重新量化結果
                            float rescale = (quant_params->input_scale * quant_params->weight_scale) / quant_params->output_scale;
                            int out_val = (int)((float)acc * rescale) + quant_params->output_zero_point;
                            
                            // 確保結果在有效範圍內
                            if (out_val < 0) out_val = 0;
                            if (out_val > 255) out_val = 255;
                            
                            output[output_idx] = (unsigned char)out_val;
                        }
                    }
                }
            }
        }
    }
}


// 從文件讀取輸入數據
unsigned char* read_input_from_file(const char* filename, 
                             int batch_size, int channels, int depth, int height, int width) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening input file: %s\n", filename);
        return NULL;
    }
    
    // 分配記憶體
    int total_size = batch_size * channels * depth * height * width;
    unsigned char* input_data = (unsigned char*)malloc(total_size * sizeof(unsigned char));
    if (input_data == NULL) {
        printf("Memory allocation failed for input data\n");
        fclose(file);
        return NULL;
    }
    
    // 讀取數據
    char line[10000]; // 假設一行不超過10000字符
    int idx = 0;
    
    // 遍歷5D數據結構
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int d = 0; d < depth; d++) {
                for (int h = 0; h < height; h++) {
                    // 讀取一行數據
                    if (fgets(line, sizeof(line), file) == NULL) {
                        printf("Error reading line at [%d][%d][%d][%d]\n", b, c, d, h);
                        free(input_data);
                        fclose(file);
                        return NULL;
                    }
                    
                    // 處理每個數字
                    char* token = strtok(line, " ,[]");
                    for (int w = 0; w < width && token != NULL; w++) {
                        int value = atoi(token);
                        int data_idx = get_5d_index(b, c, d, h, w, batch_size, channels, depth, height, width);
                        input_data[data_idx] = (unsigned char)value;
                        token = strtok(NULL, " ,[]");
                    }
                }
            }
        }
    }
    
    fclose(file);
    return input_data;
}

// 從文件讀取權重數據
char* read_weights_from_file(const char* filename, 
                         int out_channels, int in_channels, int kernel_d, int kernel_h, int kernel_w) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening weights file: %s\n", filename);
        return NULL;
    }
    
    // 分配記憶體
    int total_size = out_channels * in_channels * kernel_d * kernel_h * kernel_w;
    char* weight_data = (char*)malloc(total_size * sizeof(char));
    if (weight_data == NULL) {
        printf("Memory allocation failed for weights\n");
        fclose(file);
        return NULL;
    }
    
    // 讀取數據 (假設權重文件格式較簡單)
    int idx = 0;
    int value;
    while (idx < total_size && fscanf(file, "%d", &value) == 1) {
        weight_data[idx++] = (char)value;
    }
    
    if (idx != total_size) {
        printf("Warning: Read only %d weights out of %d expected\n", idx, total_size);
    }
    
    fclose(file);
    return weight_data;
}

// 從文件讀取偏置數據
int* read_bias_from_file(const char* filename, int out_channels) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening bias file: %s\n", filename);
        return NULL;
    }
    
    // 分配記憶體
    int* bias_data = (int*)malloc(out_channels * sizeof(int));
    if (bias_data == NULL) {
        printf("Memory allocation failed for bias\n");
        fclose(file);
        return NULL;
    }
    
    // 讀取數據
    int idx = 0;
    int value;
    while (idx < out_channels && fscanf(file, "%d", &value) == 1) {
        bias_data[idx++] = value;
    }
    
    if (idx != out_channels) {
        printf("Warning: Read only %d bias values out of %d expected\n", idx, out_channels);
    }
    
    fclose(file);
    return bias_data;
}

// 將輸出寫入文件
void write_output_to_file(const char* filename, const unsigned char* output,
                         int batch_size, int channels, int depth, int height, int width) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening output file: %s\n", filename);
        return;
    }
    
    // 寫入數據
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int d = 0; d < depth; d++) {
                for (int h = 0; h < height; h++) {
                    fprintf(file, "[");
                    for (int w = 0; w < width; w++) {
                        int idx = get_5d_index(b, c, d, h, w, batch_size, channels, depth, height, width);
                        fprintf(file, "%d", output[idx]);
                        if (w < width - 1) {
                            fprintf(file, ", ");
                        }
                    }
                    fprintf(file, "]\n");
                }
            }
        }
    }
    
    fclose(file);
}

int main() {
    // 輸入參數
    int batch_size = 1;
    int in_channels = 1;
    int in_depth = 2;  
    int in_height = 3;
    int in_width = 56;
    
    // 卷積核參數
    int kernel_d = 3;
    int kernel_h = 3;
    int kernel_w = 3;
    int out_channels = 1;
    
    // 其他參數
    int stride_d = 1, stride_h = 1, stride_w = 1;
    int padding_d = 1, padding_h = 1, padding_w = 1;
    
    // 計算輸出尺寸
    int out_depth = (in_depth + 2 * padding_d - kernel_d) / stride_d + 1;
    int out_height = (in_height + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w - kernel_w) / stride_w + 1;
    
    // 量化參數
    QuantParams quant_params = {
        .input_scale = 1.0f,
        .input_zero_point = 0,
        .weight_scale = 1.0f,
        .weight_zero_point = 0,
        .output_scale = 1.0f,
        .output_zero_point = 0
    };
    
    // 讀取輸入數據
    unsigned char* input = read_input_from_file("./input/input.txt", 
                                             batch_size, in_channels, in_depth, in_height, in_width);
    if (input == NULL) {
        return 1;
    }
    
    // 讀取權重數據
    char* weights = read_weights_from_file("./weight/conv1_dw.weight.txt", 
                                       out_channels, in_channels, kernel_d, kernel_h, kernel_w);
    if (weights == NULL) {
        free(input);
        return 1;
    }
    
    // 讀取偏置數據
    int* bias = read_bias_from_file("./weight/conv1_dw.bias.txt", out_channels);
    if (bias == NULL) {
        free(input);
        free(weights);
        return 1;
    }
    
    // 分配輸出記憶體
    unsigned char* output = (unsigned char*)malloc(batch_size * out_channels * out_depth * out_height * out_width * sizeof(unsigned char));
    if (output == NULL) {
        printf("Memory allocation failed for output\n");
        free(input);
        free(weights);
        free(bias);
        return 1;
    }
    
    // 執行3D捲積
    conv3d_depthwise_1d_array(input, weights, bias, output,
                   batch_size, in_channels, in_depth, in_height, in_width,
                   out_channels, out_depth, out_height, out_width,
                   kernel_d, kernel_h, kernel_w,
                   stride_d, stride_h, stride_w,
                   padding_d, padding_h, padding_w,
                   &quant_params);
    
    // 打印部分輸出進行驗證
    printf("輸出尺寸: %dx%dx%d\n", out_depth, out_height, out_width);
    printf("部分輸出值:\n");
    for (int d = 0; d < 1; d++) {
        for (int h = 0; h < 2; h++) {
            for (int w = 0; w < 3; w++) {
                int idx = get_5d_index(0, 0, d, h, w, batch_size, out_channels, out_depth, out_height, out_width);
                printf("output[0][0][%d][%d][%d] = %d\n", d, h, w, output[idx]);
            }
        }
    }
    
    // 將輸出寫入文件
    write_output_to_file("./C_test/output/output.txt", output, batch_size, out_channels, out_depth, out_height, out_width);
    
    // 釋放記憶體
    free(input);
    free(weights);
    free(bias);
    free(output);
    
    return 0;
}