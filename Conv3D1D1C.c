#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DIM 5
#define INPUT_DIM_1 56
#define INPUT_DIM_2 56
#define INPUT_DIM_3 16
#define INPUT_DIM_4 3
#define INPUT_DIM_5 1

int read_txt(const char* filepath, void* data_out) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        printf("無法開啟檔案：%s\n", filepath);
        return -1;
    }

    int capacity = 128;
    int size = 0;
    int* data = (int*)malloc(capacity * sizeof(int));
    if (!data) {
        printf("記憶體配置失敗\n");
        fclose(file);
        return -1;
    }

    int num;
    while (fscanf(file, "%d", &num) == 1) {
        if (size >= capacity) {
            capacity *= 2;
            int* new_data = (int*)realloc(data, capacity * sizeof(int));
            if (!new_data) {
                printf("記憶體重新配置失敗\n");
                free(data);
                fclose(file);
                return -1;
            }
            data = new_data;
        }
        data[size++] = num;
    }

    fclose(file);
    
    // 將結果存入 data_out (不論是 int** 還是 char**)
    *((int**)data_out) = data;
    
    return size;
}

int read_txt_float(const char* filepath, float** data_out) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        printf("無法開啟檔案：%s\n", filepath);
        return -1;
    }

    int capacity = 128;
    int size = 0;
    float* data = (float*)malloc(capacity * sizeof(float));
    if (!data) {
        printf("記憶體配置失敗\n");
        fclose(file);
        return -1;
    }

    float num;
    while (fscanf(file, "%f", &num) == 1) {
        if (size >= capacity) {
            capacity *= 2;
            float* new_data = (float*)realloc(data, capacity * sizeof(float));
            if (!new_data) {
                printf("記憶體重新配置失敗\n");
                free(data);
                fclose(file);
                return -1;
            }
            data = new_data;
        }
        data[size++] = num;
    }

    fclose(file);
    *data_out = data;
    return size;
}

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
                                            //printf("%d\n",acc);
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

int main() {
    
    //---------------------------- get params ----------------------
    int* input_params = NULL;
    float* quant_scale = NULL;
    int* quant_zero_point = NULL;
    
    int* conv1_dw_weight = NULL;
    int* conv1_dw_weight_scale = NULL;
    int* conv1_dw_weight_zero_point = NULL;

    int* conv1_dw_bias = NULL;
    int* conv1_dw_scale = NULL;
    int* conv1_dw_zero_point = NULL;

    int input_size = read_txt("./C_test/input/input_.txt", &input_params);
    int quant_size = read_txt_float("./C_test/weight/quant.scale.txt", &quant_scale);
    int quant_zero_point_size = read_txt("./C_test/weight/quant.zero_point.txt", &quant_zero_point);
    int weight_size = read_txt("./C_test/weight/conv1_dw.weight.txt", &conv1_dw_weight);
    int weight_scale_size = read_txt("./C_test/weight/conv1_dw.weight_scale.txt", &conv1_dw_weight_scale);
    int weight_zero_point_size = read_txt("./C_test/weight/conv1_dw.weight_zero_points.txt", &conv1_dw_weight_zero_point);
    int bias_size = read_txt("./C_test/weight/conv1_dw.bias.txt", &conv1_dw_bias);
    int bias_scale_size = read_txt("./C_test/weight/conv1_dw.scale.txt", &conv1_dw_scale);
    int bias_zero_point_size = read_txt("./C_test/weight/conv1_dw.zero_point.txt", &conv1_dw_zero_point);
    
    printf("\n部分輸入參數 (input_params):\n");
    int max_display = 100; // 最多顯示100個值
    int display_count = (input_size < max_display) ? input_size : max_display;
    
    for (int i = 0; i < display_count; i++) {
        printf("%d ", input_params[i]);
        if ((i + 1) % 10 == 0) printf("\n"); // 每10個數值換行一次
    }

    printf("\n\n部分權重參數 (conv1_dw_weight):\n");
    display_count = (weight_size < max_display) ? weight_size : max_display;
    for (int i = 0; i < display_count; i++) {
        printf("%d ", conv1_dw_weight[i]);
        if ((i + 1) % 10 == 0) printf("\n"); // 每10個數值換行一次
    }

    printf("\n\n部分量化參數 (quant_scale):\n");
    display_count = (quant_size < max_display) ? quant_size : max_display;
    for (int i = 0; i < display_count; i++) {
        printf("%f", quant_scale[i]);
        if ((i + 1) % 10 == 0) printf("\n"); // 每10個數值換行一次
    }
    printf("\n\n部分量化參數 (quant_zero_point):\n");
    display_count = (quant_zero_point_size < max_display) ? quant_zero_point_size : max_display;
    for (int i = 0; i < display_count; i++) {
        printf("%d ", quant_zero_point[i]);
        if ((i + 1) % 10 == 0) printf("\n"); // 每10個數值換行一次
    }

    printf("\n\n部分偏置參數 (conv1_dw_bias):\n");
    display_count = (bias_size < max_display) ? bias_size : max_display;
    for (int i = 0; i < display_count; i++) {
        printf("%d ", conv1_dw_bias[i]);
        if ((i + 1) % 10 == 0) printf("\n"); // 每10個數值換行一次
    }




    /*
    printf("共讀入 %d 筆資料\n", input_size);               // 3x16x56x566 = 150528
    printf("共讀入 %d 筆資料\n", quant_size);
    printf("共讀入 %d 筆資料\n", quant_zero_point_size);
    printf("共讀入 %d 筆資料\n", weight_size);              // 1x3x3x3x3 = 27
    printf("共讀入 %d 筆資料\n", weight_scale_size);
    printf("共讀入 %d 筆資料\n", weight_zero_point_size);
    printf("共讀入 %d 筆資料\n", bias_size);
    printf("共讀入 %d 筆資料\n", bias_scale_size);
    printf("共讀入 %d 筆資料\n", bias_zero_point_size);*/
    //========================== get params ==========================



    

    unsigned char* input_data = (unsigned char*)malloc(input_size * sizeof(unsigned char));
    char* weight_data = (char*)malloc(weight_size * sizeof(char));

    // 將 int 類型資料轉換為需要的類型
    for (int i = 0; i < input_size; i++) {
        input_data[i] = (unsigned char)input_params[i];
    }
    for (int i = 0; i < weight_size; i++) {
        weight_data[i] = (char)conv1_dw_weight[i];
    }

    // 計算輸出維度（根據卷積、步長和填充）
    int out_depth = (INPUT_DIM_3 + 2*1 - 3) / 1 + 1;  // (in_depth + 2*padding_d - kernel_d) / stride_d + 1
    int out_height = (INPUT_DIM_2 + 2*1 - 3) / 1 + 1; // (in_height + 2*padding_h - kernel_h) / stride_h + 1
    int out_width = (INPUT_DIM_1 + 2*1 - 3) / 1 + 1;  // (in_width + 2*padding_w - kernel_w) / stride_w + 1

    // 分配輸出記憶體
    unsigned char* conv1_dw_output = (unsigned char*)malloc(
        INPUT_DIM_5 * INPUT_DIM_4 * out_depth * out_height * out_width * sizeof(unsigned char)
    );

    QuantParams quant_params = {
        .input_scale = (float)quant_scale[0],         // 假設需要除以1000來獲得正確的浮點數值
        .input_zero_point = quant_zero_point[0],
        .weight_scale = (float)conv1_dw_weight_scale[0],  // 假設需要除以1000來獲得正確的浮點數值
        .weight_zero_point = conv1_dw_weight_zero_point[0],
        .output_scale = (float)conv1_dw_scale[0],         // 假設需要除以1000來獲得正確的浮點數值
        .output_zero_point = conv1_dw_zero_point[0]
    };

    conv3d_depthwise_1d_array(
        input_data,  // 已轉換為 unsigned char*
        weight_data, // 已轉換為 char*
        conv1_dw_bias,
        conv1_dw_output,
        INPUT_DIM_5, INPUT_DIM_4, INPUT_DIM_3, INPUT_DIM_2, INPUT_DIM_1,
        out_depth, out_height, out_width,
        3, 3, 3,     // 核心尺寸
        1, 1, 1,     // 步長
        1, 1, 1,     // 填充
        1,           // depth_multiplier
        &quant_params // 量化參數
    );


    // ----------------------------- print output ----------------------
    printf("\n\n輸出尺寸: %dx%dx%dx%dx%d\n", INPUT_DIM_5, INPUT_DIM_4, out_depth, out_height, out_width);
    
    // 將所有輸出值儲存到檔案
    FILE* output_file = fopen("./C_test/test_output/conv1_dw_output.txt", "w");
    if (!output_file) {
        printf("無法建立輸出檔案\n");
        return -1;
    }
    
    // 寫入輸出尺寸信息
    fprintf(output_file, "# 輸出尺寸: %dx%dx%dx%dx%d\n", INPUT_DIM_5, INPUT_DIM_4, out_depth, out_height, out_width);
    
    // 儲存所有輸出值
    int total_elements = INPUT_DIM_5 * INPUT_DIM_4 * out_depth * out_height * out_width;
    printf("正在儲存 %d 個輸出元素到檔案...\n", total_elements);
    
    // 按照5D格式儲存所有元素
    for (int b = 0; b < INPUT_DIM_5; b++) {
        for (int c = 0; c < INPUT_DIM_4; c++) {
            for (int d = 0; d < out_depth; d++) {
                for (int h = 0; h < out_height; h++) {
                    for (int w = 0; w < out_width; w++) {
                        int idx = get_5d_index(b, c, d, h, w, INPUT_DIM_5, INPUT_DIM_4, out_depth, out_height, out_width);
                        fprintf(output_file, "%d ", conv1_dw_output[idx]);
                    }
                    fprintf(output_file, "\n"); // 每行結束
                }
                fprintf(output_file, "\n"); // 每個高度層結束
            }
            fprintf(output_file, "\n"); // 每個深度層結束
        }
        fprintf(output_file, "\n"); // 每個通道結束
    }
    
    fclose(output_file);
    printf("所有輸出值已儲存到 ./C_test/test_output/conv1_dw_output.txt\n");
    


    // ----------------------------- free memory ----------------------
    free(input_params);
    free(quant_scale);
    free(quant_zero_point);
    free(conv1_dw_weight);
    free(conv1_dw_weight_scale);
    free(conv1_dw_weight_zero_point);
    free(conv1_dw_bias);
    free(conv1_dw_scale);
    free(conv1_dw_zero_point);
    return 0;
}