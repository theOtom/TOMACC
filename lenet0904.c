#include<ap_cint.h>
#include <ap_fixed.h>
#include<math.h>

typedef ap_fixed<16,11> fixed16;

#define IN_DATASIZE 196
#define OUT_DATASIZE 3

#define C1_OCH 20
#define C1_OSIZE 24
#define C1_ICH 1
#define C1_ISIZE 28
#define C1_K 5

#define P1_OSIZE 12
#define P1_K 2

#define C2_OCH 50
#define C2_OSIZE 8
#define C2_ICH 20
#define C2_ISIZE 12
#define C2_K 5
#define C2_P 10

#define P2_OSIZE 4
#define P2_K 2

#define F1_N 500
#define F1_M 800
#define F1_P 10

#define F2_N 10
#define F2_M 500
#define F2_P 10

#define RESULTSIZE 12

#define ALL_WB_SIZE (520+25050+400500+5010)

//void rx32(
//		uint169 input[IN_DATASIZE],
//		float output[C1_ICH][C1_ISIZE][C1_ISIZE]
//		){
//#pragma HLS INLINE off
//	int i, j, k;
//	float input_tmp[C1_ICH*C1_ISIZE*C1_ISIZE];
//	for (i = 0; i < IN_DATASIZE; i++) {
//		input_tmp[i*4+0] = input[i] >> 0;
//		input_tmp[i*4+1] = input[i] >> 32;
//		input_tmp[i*4+2] = input[i] >> 64;
//		input_tmp[i*4+3] = input[i] >> 96;
//	}
//	for(i = 0; i < C1_ICH; i++) {
//		for(j = 0; j < C1_ISIZE; j++) {
//			for(k = 0; k < C1_ISIZE; k++) {
//				output[i][j][k] = input_tmp[i*C1_ISIZE*C1_ISIZE + j*C1_ISIZE + k];
//			}
//		}
//	}
//	return;
//}

void load_input(
		float input[C1_ICH*C1_ISIZE*C1_ISIZE],
		float output[C1_ICH][C1_ISIZE][C1_ISIZE]
		){
#pragma HLS INLINE off
	int i, j, k;
	float tmp[C1_ICH*C1_ISIZE*C1_ISIZE];

	for (i = 0; i < C1_ICH*C1_ISIZE*C1_ISIZE; i++) {
		tmp[i] = input[i];
	}

	for(i = 0; i < C1_ICH; i++) {
		for(j = 0; j < C1_ISIZE; j++) {
			for(k = 0; k < C1_ISIZE; k++) {
				output[i][j][k] = tmp[i*C1_ISIZE*C1_ISIZE + j*C1_ISIZE + k];
			}
		}
	}
	return;
}

void load_wb(
		float input[ALL_WB_SIZE],
		float conv1_w[C1_OCH][C1_ICH][C1_K][C1_K],
		float conv1_b[C1_OCH],
		float conv2_w[C2_OCH][C2_ICH][C2_K][C2_K],
		float conv2_b[C2_OCH],
		float fc1_w[F1_N][F1_M],
		float fc1_b[F1_N],
		float fc2_w[F2_N][F2_M],
		float fc2_b[F2_N]
		){
#pragma HLS INLINE off
	int i, j, k, l;
	unsigned long datasize, offset;
	//datasize = (unsigned long)input[0];
	//printf("HLS: datasize=%lu\n", datasize);

//	float tmp[ALL_WB_SIZE];
//
//	for (i = 0; i < ALL_WB_SIZE; i++) {
//		tmp[i] = input[i];
//	}

	//CONV1_WB
	offset = 0;
	for(i = 0; i < C1_OCH; i++) {
		for(j = 0; j < C1_ICH; j++) {
			for(k = 0; k < C1_K; k++) {
				for (l = 0; l < C1_K; l++) {
					conv1_w[i][j][k][l] = input[offset + i*C1_ICH*C1_K*C1_K + j*C1_K*C1_K + k*C1_K + l];
				}
			}
		}
	}
	for(i = 0; i < C1_OCH; i++) {
		conv1_b[i] = input[offset + C1_OCH*C1_ICH*C1_K*C1_K + i];
	}

	//CONV2_WB
	offset = 520;
	for(i = 0; i < C2_OCH; i++) {
		for(j = 0; j < C2_ICH; j++) {
			for(k = 0; k < C2_K; k++) {
				for (l = 0; l < C2_K; l++) {
					conv2_w[i][j][k][l] = input[offset + i*C2_ICH*C2_K*C2_K + j*C2_K*C2_K + k*C2_K + l];
				}
			}
		}
	}
	for(i = 0; i < C2_OCH; i++) {
		conv2_b[i] = input[offset + C2_OCH*C2_ICH*C2_K*C2_K + i];
	}

	//FC1_WB
	offset = 520 + 25050;
	for(i = 0; i < F1_N; i++)
		for (j = 0; j < F1_M; j++)
			fc1_w[i][j] = input[offset + i*F1_M + j];
	for(i = 0; i < F1_N; i++)
		fc1_b[i] = input[offset + F1_N*F1_M + i];

	//FC1_WB
	offset = 520 + 25050 + 400500;
	for(i = 0; i < F2_N; i++)
		for (j = 0; j < F2_M; j++)
			fc2_w[i][j] = input[offset + i*F2_M + j];
	for(i = 0; i < F2_N; i++)
		fc2_b[i] = input[offset + F2_N*F2_M + i];

	return;
}

void conv1(
		float input[C1_ICH][C1_ISIZE][C1_ISIZE],
		float weight[C1_OCH][C1_ICH][C1_K][C1_K],
		float bias[C1_OCH],
		float output[C1_OCH][C1_OSIZE][C1_OSIZE]
){
#pragma HLS INLINE off
	int ox, oy, kx, ky, n, m;
	static int stride = 1;

	//Calculate
		for (ox = 0; ox < C1_OSIZE; ox++) {
			for (oy = 0; oy < C1_OSIZE; oy++) {
				for (n = 0; n < C1_OCH; n++) {
					output[n][ox][oy] = bias[n];
					for (m = 0; m < C1_ICH; m++) {
						for (kx = 0; kx < C1_K; kx++) {
							for (ky = 0; ky < C1_K; ky++) {

								//ix[kx] = stride*ox+kx;
								//iy[ky] = stride*oy+ky;
								output[n][ox][oy] +=
										weight[n][m][kx][ky] *
										input[m][stride*ox+kx][stride*oy+ky];

							}
						}
					}
				}
			}
		}
	return;
}

void pool1(
		float input[C1_OCH][C1_OSIZE][C1_OSIZE],
		float output[C1_OCH][P1_OSIZE][P1_OSIZE]
		){
#pragma HLS INLINE off
	int ox, oy, kx, ky, ix, iy, n, m;
	float tmp, max;

	int stride = 2;
	  for (n = 0; n < C1_OCH; n++) {
		for (ox = 0; ox < P1_OSIZE; ox++) {
		  for (oy = 0; oy < P1_OSIZE; oy++) {
			max = -256.0;
			for (kx = 0; kx < P1_K; kx++) {
			  for (ky = 0; ky < P1_K; ky++) {
				 tmp = input[n][ox*stride+kx][oy*stride+ky];
				//tmp = *(input+och*isize*isize+krow*isize+kcol+(orow*isize*stride+ocol*stride));
				if (max < tmp) max = tmp;
			  }
			}
			output[n][ox][oy] = max;
			//*(output+och*osize*osize+osize*orow+ocol) = max;
		  }
		}
	  }
	return;
}

void conv2(
		float input[C2_ICH][C2_ISIZE][C2_ISIZE],
		float weight[C2_OCH][C2_ICH][C2_K][C2_K],
		float bias[C2_OCH],
		float output[C2_OCH][C2_OSIZE][C2_OSIZE]
){
#pragma HLS INLINE off
	int ox, oy, kx, ky, n, m;
	static int stride = 1;

	//Calculate
		for (ox = 0; ox < C2_OSIZE; ox++) {
			for (oy = 0; oy < C2_OSIZE; oy++) {
				for (n = 0; n < C2_OCH; n++) {
					output[n][ox][oy] = bias[n];
					for (m = 0; m < C2_ICH; m++) {
						for (kx = 0; kx < C2_K; kx++) {
							for (ky = 0; ky < C2_K; ky++) {
								//ix[kx] = stride*ox+kx;
								//iy[ky] = stride*oy+ky;
								output[n][ox][oy] +=
										weight[n][m][kx][ky] *
										input[m][stride*ox+kx][stride*oy+ky];

							}
						}
					}
				}
			}
		}
	return;
}

void pool2(
		float input[C2_OCH][C2_OSIZE][C2_OSIZE],
		float output[C2_OCH][P2_OSIZE][P2_OSIZE]
		){
#pragma HLS INLINE off
	int ox, oy, kx, ky, ix, iy, n, m;
	float tmp, max;

	int stride = 2;
	  for (n = 0; n < C2_OCH; n++) {
		for (ox = 0; ox < P2_OSIZE; ox++) {
		  for (oy = 0; oy < P2_OSIZE; oy++) {
			max = -256.0;
			for (kx = 0; kx < P2_K; kx++) {
			  for (ky = 0; ky < P2_K; ky++) {
				 tmp = input[n][ox*stride+kx][oy*stride+ky];
				//tmp = *(input+och*isize*isize+krow*isize+kcol+(orow*isize*stride+ocol*stride));
				if (max < tmp) max = tmp;
			  }
			}
			output[n][ox][oy] = max;
			//*(output+och*osize*osize+osize*orow+ocol) = max;
		  }
		}
	  }
	return;
}

void flatten(float input[C2_OCH][P2_OSIZE][P2_OSIZE], float output[F1_M]){
#pragma HLS INLINE off
	int ox, oy, n;
	for (n = 0; n < C2_OCH; n++) {
		for (ox = 0; ox < P2_OSIZE; ox++) {
		 	for (oy = 0; oy < P2_OSIZE; oy++) {
		 		output[n*P2_OSIZE*P2_OSIZE + ox*P2_OSIZE + oy] = input[n][ox][oy];
		 	}
		}
	}
	return;
}

void fc1(float input[F1_M], float weight[F1_N][F1_M], float bias[F1_N], float output[F1_N]) {
#pragma HLS INLINE off
	int i, j, p;
	for (i = 0; i < F1_N; i++) {
		output[i] = bias[i];
		for (j = 0; j < F1_M; j+=F1_P) {
			for (p = 0; p < F1_P; p++) {
#pragma HLS UNROLL
				output[i] += input[j+p] * weight[i][j+p];
			}
		}
		if (output[i] < 0.0) output[i] = 0.0;
	}
	return;
}

void fc2(float input[F2_M], float weight[F2_N][F2_M], float bias[F2_N], float output[F2_N]) {
#pragma HLS INLINE off
	int i, j, k, l, p;
	float sum = 0.0;
	float output_tmp[F2_N];

	for (i = 0; i < F2_N; i++) {
		output_tmp[i] = bias[i];
		for (j = 0; j < F2_M; j+=F2_P) {
			for (p = 0; p < F2_P; p++) {
#pragma HLS UNROLL
				output_tmp[i] += input[j+p] * weight[i][j+p];
			}
		}
	}
	for (k = 0; k < F2_N; k++) {
		sum += expf(output_tmp[k]);
	}
	for (l = 0; l < F2_N; l++) {
		output[l] = expf(output_tmp[l]) / sum;
	}
	return;
}

void store_output(float input[F2_N], float output[RESULTSIZE]){
#pragma HLS INLINE off
	int i;
	for(i = 0; i < F2_N; i++)
		output[i] = input[i];
	for(i = F2_N; i < RESULTSIZE; i++)
		output[i] = 0.0;
	return;
}

void lenet0904(
		float input[C1_ICH*C1_ISIZE*C1_ISIZE],
		float output[RESULTSIZE],
		//uint169 input[IN_DATASIZE],
		//uint169 output[3],
		float wb[ALL_WB_SIZE]
){
//#pragma HLS dataflow
#pragma HLS INTERFACE axis register both port=input
#pragma HLS INTERFACE axis register both port=output
#pragma HLS INTERFACE axis register both port=wb
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=2

	//static float input_tmp[C1_ICH][C1_ISIZE][C1_ISIZE];

	static float image[C1_ICH][C1_ISIZE][C1_ISIZE];

	static float conv1_w[C1_OCH][C1_ICH][C1_K][C1_K];
	static float conv1_b[C1_OCH];
	static float conv1_out[C1_OCH][C1_OSIZE][C1_OSIZE];

	static float pool1_out[C1_OCH][P1_OSIZE][P1_OSIZE];

	static float conv2_w[C2_OCH][C2_ICH][C2_K][C2_K];
	static float conv2_b[C2_OCH];
	static float conv2_out[C2_OCH][C2_OSIZE][C2_OSIZE];

	static float pool2_out[C2_OCH][P2_OSIZE][P2_OSIZE];

	static float flat_out[F1_M];

	static float fc1_w[F1_N][F1_M];
	static float fc1_b[F1_N];
	static float fc1_out[F1_N];

	static float fc2_w[F2_N][F2_M];
	static float fc2_b[F2_N];
	static float fc2_out[F2_N];

	static int wb_flag = 0;

	if (wb_flag == 0) {
		load_wb(wb, conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b);
		wb_flag = 1;
	}

	load_input(input, image);

	conv1(image, conv1_w, conv1_b, conv1_out);
	pool1(conv1_out, pool1_out);

	conv2(pool1_out, conv2_w, conv2_b, conv2_out);
	pool2(conv2_out, pool2_out);

	flatten(pool2_out, flat_out);

	fc1(flat_out, fc1_w, fc1_b, fc1_out);

	fc2(fc1_out, fc2_w, fc2_b, fc2_out);

	store_output(fc2_out, output);

	return;
}

