#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
 
#define IMAGEFILE "./dat/image_0.dat"
#define CHECK_PARAMS 0

#define IMAGE_SIZE 3*227*227
#define CONV1_W_SIZE 96*3*11*11
#define CONV1_B_SIZE 96
#define CONV1_OUT_SIZE 96*55*55
#define NORM1_OUT_SIZE 96*55*55
#define POOL1_OUT_SIZE 96*27*27

#define PAD2_OUT_SIZE 96*31*31
#define CONV2_W_SIZE 256*96*5*5
#define CONV2_B_SIZE 256
#define CONV2_OUT_SIZE 256*27*27
#define NORM2_OUT_SIZE 256*27*27
#define POOL2_OUT_SIZE 256*13*13

#define PAD3_OUT_SIZE 256*15*15
#define CONV3_W_SIZE 384*256*3*3
#define CONV3_B_SIZE 384
#define CONV3_OUT_SIZE 384*13*13

#define PAD4_OUT_SIZE 384*15*15
#define CONV4_W_SIZE 384*384*3*3
#define CONV4_B_SIZE 384
#define CONV4_OUT_SIZE 384*13*13

#define PAD5_OUT_SIZE 384*15*15
#define CONV5_W_SIZE 256*384*3*3
#define CONV5_B_SIZE 256
#define CONV5_OUT_SIZE 256*13*13
#define POOL5_OUT_SIZE 256*6*6

#define FC6_W_SIZE 4096*9216
#define FC6_B_SIZE 4096
#define FC6_OUT_SIZE 4096

#define FC7_W_SIZE 4096*4096
#define FC7_B_SIZE 4096
#define FC7_OUT_SIZE 4096

#define FC8_W_SIZE 1000*4096
#define FC8_B_SIZE 1000
#define FC8_OUT_SIZE 1000

/* int16_t float2fix4_12(float f) */
/* { */
/*   int16_t out; */
/*   out = (int16_t)(f * 4096); */
/*   return out; */
/* } */

int16_t clip(float f)
{
  int16_t fix;
  if (f > 7.999755859375) { //0x7FFF 
	fix = 0x7FFF;
  } else if (f < -8.0 ) { //0x8000
	fix = 0x8000;
  } else {
	fix = (int16_t)(f * 4096);
  }

  // printf("%f -> %04X\n", f, fix & 0xFFFF);
  return fix;
}

void clip_array(float *data, int size)
{
  int i;

  for (i = 0; i < size; i++) {
	if (data[i] > 7.999755859375) {
	  data[i] = 7.999755859375;
	} else if (data[i] < -8.0 ) {
	  data[i] = -8.0;
	}
  }
  
}

void make_fix_header(float *data, int size, char *name)
{
  FILE *fp;
  int i;
  char buf[1024];

  sprintf(buf, "%s.h", name);
  if ((fp = fopen(buf, "w+")) == NULL) {
	perror("open: ");
	return ;
  }
  fprintf(fp, "int16_t %s[%d] = {\n", name, size);

  for (i = 0; i < size; i++) {
	fprintf(fp, "\t0x%04X", clip(data[i]) & 0xFFFF);
	if ((i % 8 == 7) && (i != size - 1)) {
	  fprintf(fp, ",\n");
	} else if (i == size - 1) {
	  fprintf(fp, "\n};\n");
	} else {
	  fprintf(fp, ", ");
	}
  }
}


int main(){
  int i, j, k, l;
  
  float *image;
  float *conv1_w, *conv1_b, *conv1_out;
  float *norm1_out;
  float *pool1_out;

  float *pad2_out;
  float *conv2_w, *conv2_b, *conv2_out;
  float *norm2_out;
  float *pool2_out;

  float *pad3_out;
  float *conv3_w, *conv3_b, *conv3_out;

  float *pad4_out;
  float *conv4_w, *conv4_b, *conv4_out;

  float *pad5_out;
  float *conv5_w, *conv5_b, *conv5_out;
  float *pool5_out;

  float *fc6_w, *fc6_b, *fc6_out;
  float *fc7_w, *fc7_b, *fc7_out;
  float *fc8_w, *fc8_b, *fc8_out;

  float *debug;

  printf("/// ImageNet ///\n\n");fflush(stdout);
  
  printf("Memory allocation ...\n");fflush(stdout);
  if ((image = (float *)malloc(sizeof(float)*IMAGE_SIZE)) == NULL ||
	  (conv1_w = (float *)malloc(sizeof(float)*CONV1_W_SIZE)) == NULL ||
	  (conv1_b = (float *)malloc(sizeof(float)*CONV1_B_SIZE)) == NULL ||
	  (conv1_out = (float *)malloc(sizeof(float)*CONV1_OUT_SIZE)) == NULL ||
	  (norm1_out = (float *)malloc(sizeof(float)*NORM1_OUT_SIZE)) == NULL ||
	  (pool1_out = (float *)malloc(sizeof(float)*POOL1_OUT_SIZE)) == NULL ||

	  (pad2_out = (float *)malloc(sizeof(float)*PAD2_OUT_SIZE)) == NULL ||
	  (conv2_w = (float *)malloc(sizeof(float)*CONV2_W_SIZE)) == NULL ||
	  (conv2_b = (float *)malloc(sizeof(float)*CONV2_B_SIZE)) == NULL ||
	  (conv2_out = (float *)malloc(sizeof(float)*CONV2_OUT_SIZE)) == NULL ||
	  (norm2_out = (float *)malloc(sizeof(float)*NORM2_OUT_SIZE)) == NULL ||
	  (pool2_out = (float *)malloc(sizeof(float)*POOL2_OUT_SIZE)) == NULL ||

	  (pad3_out = (float *)malloc(sizeof(float)*PAD3_OUT_SIZE)) == NULL ||
	  (conv3_w = (float *)malloc(sizeof(float)*CONV3_W_SIZE)) == NULL ||
	  (conv3_b = (float *)malloc(sizeof(float)*CONV3_B_SIZE)) == NULL ||
	  (conv3_out = (float *)malloc(sizeof(float)*CONV3_OUT_SIZE)) == NULL ||

	  (pad4_out = (float *)malloc(sizeof(float)*PAD4_OUT_SIZE)) == NULL ||
	  (conv4_w = (float *)malloc(sizeof(float)*CONV4_W_SIZE)) == NULL ||
	  (conv4_b = (float *)malloc(sizeof(float)*CONV4_B_SIZE)) == NULL ||
	  (conv4_out = (float *)malloc(sizeof(float)*CONV4_OUT_SIZE)) == NULL ||

	  (pad5_out = (float *)malloc(sizeof(float)*PAD5_OUT_SIZE)) == NULL ||
	  (conv5_w = (float *)malloc(sizeof(float)*CONV5_W_SIZE)) == NULL ||
	  (conv5_b = (float *)malloc(sizeof(float)*CONV5_B_SIZE)) == NULL ||
	  (conv5_out = (float *)malloc(sizeof(float)*CONV5_OUT_SIZE)) == NULL ||
	  (pool5_out = (float *)malloc(sizeof(float)*POOL5_OUT_SIZE)) == NULL ||

	  (fc6_w = (float *)malloc(sizeof(float)*FC6_W_SIZE)) == NULL ||
	  (fc6_b = (float *)malloc(sizeof(float)*FC6_B_SIZE)) == NULL ||
	  (fc6_out = (float *)malloc(sizeof(float)*FC6_OUT_SIZE)) == NULL ||
	  
	  (fc7_w = (float *)malloc(sizeof(float)*FC7_W_SIZE)) == NULL ||
	  (fc7_b = (float *)malloc(sizeof(float)*FC7_B_SIZE)) == NULL ||
	  (fc7_out = (float *)malloc(sizeof(float)*FC7_OUT_SIZE)) == NULL ||
	  
	  (fc8_w = (float *)malloc(sizeof(float)*FC8_W_SIZE)) == NULL ||
	  (fc8_b = (float *)malloc(sizeof(float)*FC8_B_SIZE)) == NULL ||
	  (fc8_out = (float *)malloc(sizeof(float)*FC8_OUT_SIZE)) == NULL ||
	  0) {
	printf("MemError\n");
	exit(1);
  }
  printf("\n");

  printf("Read params ...\n\n");fflush(stdout);
  //Read image data
  read_binary(IMAGEFILE, image, IMAGE_SIZE);
  print_params("IMAGE : ", image, IMAGE_SIZE);
  //Read CONV1 params
  read_binary("./dat/conv1_0.dat", conv1_w, CONV1_W_SIZE);
  print_params("CONV1_W : ", conv1_w, CONV1_W_SIZE);
  read_binary("./dat/conv1_1.dat", conv1_b, CONV1_B_SIZE);
  print_params("CONV1_B : ", conv1_b, CONV1_B_SIZE);
  //Read CONV2 params
  read_binary("./dat/conv2_0.dat", conv2_w, CONV2_W_SIZE);
  print_params("CONV2_W : ", conv2_w, CONV2_W_SIZE);
  read_binary("./dat/conv2_1.dat", conv2_b, CONV2_B_SIZE);
  print_params("CONV2_B : ", conv2_b, CONV2_B_SIZE);
  //Read CONV3 params
  read_binary("./dat/conv3_0.dat", conv3_w, CONV3_W_SIZE);
  print_params("CONV3_W : ", conv3_w, CONV3_W_SIZE);
  read_binary("./dat/conv3_1.dat", conv3_b, CONV3_B_SIZE);
  print_params("CONV3_B : ", conv3_b, CONV3_B_SIZE);
  //Read CONV4 params
  read_binary("./dat/conv4_0.dat", conv4_w, CONV4_W_SIZE);
  print_params("CONV4_W : ", conv4_w, CONV4_W_SIZE);
  read_binary("./dat/conv4_1.dat", conv4_b, CONV4_B_SIZE);
  print_params("CONV4_B : ", conv4_b, CONV4_B_SIZE);
  //Read CONV5 params
  read_binary("./dat/conv5_0.dat", conv5_w, CONV5_W_SIZE);
  print_params("CONV5_W : ", conv5_w, CONV5_W_SIZE);
  read_binary("./dat/conv5_1.dat", conv5_b, CONV5_B_SIZE);
  print_params("CONV5_B : ", conv5_b, CONV5_B_SIZE);
  //Read FC6 params
  read_binary("./dat/fc6_0.dat", fc6_w, FC6_W_SIZE);
  print_params("FC6_W : ", fc6_w, FC6_W_SIZE);
  read_binary("./dat/fc6_1.dat", fc6_b, FC6_B_SIZE);
  print_params("FC6_B : ", fc6_b, FC6_B_SIZE);
  //Read FC7 params
  read_binary("./dat/fc7_0.dat", fc7_w, FC7_W_SIZE);
  print_params("FC7_W : ", fc7_w, FC7_W_SIZE);
  read_binary("./dat/fc7_1.dat", fc7_b, FC7_B_SIZE);
  print_params("FC7_B : ", fc7_b, FC7_B_SIZE);
  //Read FC8 params
  read_binary("./dat/fc8_0.dat", fc8_w, FC8_W_SIZE);
  print_params("FC8_W : ", fc8_w, FC8_W_SIZE);
  read_binary("./dat/fc8_1.dat", fc8_b, FC8_B_SIZE);
  print_params("FC8_B : ", fc8_b, FC8_B_SIZE);

  /*
  //USE TEXTFILE
  //Read image data
  read_params(IMAGEFILE, image, IMAGE_SIZE);
  print_params("IMAGE : ", image, IMAGE_SIZE);
  //Read CONV1 params
  read_params("./txt/conv1_0.txt", conv1_w, CONV1_W_SIZE);
  print_params("CONV1_W : ", conv1_w, CONV1_W_SIZE);
  read_params("./txt/conv1_1.txt", conv1_b, CONV1_B_SIZE);
  print_params("CONV1_B : ", conv1_b, CONV1_B_SIZE);
  //Read CONV2 params
  read_params("./txt/conv2_0.txt", conv2_w, CONV2_W_SIZE);
  print_params("CONV2_W : ", conv2_w, CONV2_W_SIZE);
  read_params("./txt/conv2_1.txt", conv2_b, CONV2_B_SIZE);
  print_params("CONV2_B : ", conv2_b, CONV2_B_SIZE);
  //Read CONV3 params
  read_params("./txt/conv3_0.txt", conv3_w, CONV3_W_SIZE);
  print_params("CONV3_W : ", conv3_w, CONV3_W_SIZE);
  read_params("./txt/conv3_1.txt", conv3_b, CONV3_B_SIZE);
  print_params("CONV3_B : ", conv3_b, CONV3_B_SIZE);
  //Read CONV4 params
  read_params("./txt/conv4_0.txt", conv4_w, CONV4_W_SIZE);
  print_params("CONV4_W : ", conv4_w, CONV4_W_SIZE);
  read_params("./txt/conv4_1.txt", conv4_b, CONV4_B_SIZE);
  print_params("CONV4_B : ", conv4_b, CONV4_B_SIZE);
  //Read CONV5 params
  read_params("./txt/conv5_0.txt", conv5_w, CONV5_W_SIZE);
  print_params("CONV5_W : ", conv5_w, CONV5_W_SIZE);
  read_params("./txt/conv5_1.txt", conv5_b, CONV5_B_SIZE);
  print_params("CONV5_B : ", conv5_b, CONV5_B_SIZE);
  //Read FC6 params
  read_params("./txt/fc6_0.txt", fc6_w, FC6_W_SIZE);
  print_params("FC6_W : ", fc6_w, FC6_W_SIZE);
  read_params("./txt/fc6_1.txt", fc6_b, FC6_B_SIZE);
  print_params("FC6_B : ", fc6_b, FC6_B_SIZE);
  //Read FC7 params
  read_params("./txt/fc7_0.txt", fc7_w, FC7_W_SIZE);
  print_params("FC7_W : ", fc7_w, FC7_W_SIZE);
  read_params("./txt/fc7_1.txt", fc7_b, FC7_B_SIZE);
  print_params("FC7_B : ", fc7_b, FC7_B_SIZE);
  //Read FC8 params
  read_params("./txt/fc8_0.txt", fc8_w, FC8_W_SIZE);
  print_params("FC8_W : ", fc8_w, FC8_W_SIZE);
  read_params("./txt/fc8_1.txt", fc8_b, FC8_B_SIZE);
  print_params("FC8_B : ", fc8_b, FC8_B_SIZE)
	*/

  printf("\n");


  //FEED-FORWARD
  printf("Feed forward ...\n\n");fflush(stdout);
//void convolution(float *input, int isize, int ichan, float *output, int osize, int ochan, float *weight, float *bias, int ksize, int stride)
  convolution(image, 227, 3, conv1_out, 55, 96, conv1_w, conv1_b, 11, 4);//CONV1
  relu(conv1_out, 55, 96);

  lrn(conv1_out, 55, 96, norm1_out, 1, 5, 0.0001, 0.75);//NORM1

  maxpooling(norm1_out, 55, 96, pool1_out, 27, 3, 2);//POOL1

  padding(pool1_out, 27, 96, pad2_out, 2);//PAD2
  convolution(pad2_out, 31, 96, conv2_out, 27, 256, conv2_w, conv2_b, 5, 1);//CONV2
  relu(conv2_out, 27, 256);

  lrn(conv2_out, 27, 256, norm2_out, 1, 5, 0.0001, 0.75);//NORM2
  
  maxpooling(norm2_out, 27, 256, pool2_out, 13, 3, 2);//POOL2

  padding(pool2_out, 13, 256, pad3_out, 1);//PAD3
  convolution(pad3_out, 15, 256, conv3_out, 13, 384, conv3_w, conv3_b, 3, 1);//CONV3
  relu(conv3_out, 13, 384);

  padding(conv3_out, 13, 384, pad4_out, 1);//PAD4
  convolution(pad4_out, 15, 384, conv4_out, 13, 384, conv4_w, conv4_b, 3, 1);//CONV4
  relu(conv4_out, 13, 384);

  padding(conv4_out, 13, 384, pad5_out, 1);//PAD5
  convolution(pad5_out, 15, 384, conv5_out, 13, 256, conv5_w, conv5_b, 3, 1);//CONV5
  relu(conv5_out, 13, 256);

  maxpooling(conv5_out, 13, 256, pool5_out, 6, 3, 2);//POOL5

  classifier(pool5_out, 9216, fc6_out, 4096, fc6_w, fc6_b);//FC6
  relu(fc6_out, 1, 4096);

  make_fix_header(fc6_out, FC6_OUT_SIZE, "fc6_out");
  clip_array(fc6_out, FC6_OUT_SIZE);
  make_fix_header(fc7_w, FC7_W_SIZE, "fc7_w");
  clip_array(fc7_w, FC7_W_SIZE);
  make_fix_header(fc7_b, FC7_B_SIZE, "fc7_b");
  clip_array(fc7_b, FC7_B_SIZE);

  int count = 0;
  for (i = 0; i < FC6_OUT_SIZE; i++) {
	if (fc6_out[i] > 8 || fc6_out[i] < -8) {
	  count++;
	}
  }

  printf("count %d\n", count);
  classifier(fc6_out, 4096, fc7_out, 4096, fc7_w, fc7_b);//FC7

  make_fix_header(fc7_out, FC7_OUT_SIZE, "fc7_out_true");
  clip_array(fc7_out, FC7_OUT_SIZE);

  relu(fc7_out, 1, 4096);

  make_fix_header(fc7_out, FC7_OUT_SIZE, "fc7_relu_out");
  clip_array(fc7_out, FC7_OUT_SIZE);

  classifier(fc7_out, 4096, fc8_out, 1000, fc8_w, fc8_b);//FC8
  softmax(fc8_out, 1000);

  show_result(fc8_out, "./txt/category.txt", FC8_OUT_SIZE);

  //Compare between my outputs and caffe's outputs
  if (CHECK_PARAMS) {
	printf("Check params ...\n\n");fflush(stdout);

	print_params("CONV1_OUT : ", conv1_out, CONV1_OUT_SIZE);
	check_params(conv1_out, "./txt/conv1_out.txt", CONV1_OUT_SIZE);

	print_params("NORM1_OUT : ", norm1_out, NORM1_OUT_SIZE);
	check_params(norm1_out, "./txt/norm1_out.txt", NORM1_OUT_SIZE);

	print_params("POOL1_OUT : ", pool1_out, POOL1_OUT_SIZE);
	check_params(pool1_out, "./txt/pool1_out.txt", POOL1_OUT_SIZE);

	print_params("CONV2_OUT : ", conv2_out, CONV2_OUT_SIZE);
	check_params(conv2_out, "./txt/conv2_out.txt", CONV2_OUT_SIZE);

	print_params("NORM2_OUT : ", conv2_out, NORM2_OUT_SIZE);
	check_params(norm2_out, "./txt/norm2_out.txt", NORM2_OUT_SIZE);

	print_params("POOL2_OUT : ", pool2_out, POOL2_OUT_SIZE);
	check_params(pool2_out, "./txt/pool2_out.txt", POOL2_OUT_SIZE);

	print_params("CONV3_OUT : ", conv3_out, CONV3_OUT_SIZE);
	check_params(conv3_out, "./txt/conv3_out.txt", CONV3_OUT_SIZE);

	print_params("CONV4_OUT : ", conv4_out, CONV4_OUT_SIZE);
	check_params(conv4_out, "./txt/conv4_out.txt", CONV4_OUT_SIZE);

	print_params("CONV5_OUT : ", conv5_out, CONV5_OUT_SIZE);
	check_params(conv5_out, "./txt/conv5_out.txt", CONV5_OUT_SIZE);

	print_params("POOL5_OUT : ", pool5_out, POOL5_OUT_SIZE);
	check_params(pool5_out, "./txt/pool5_out.txt", POOL5_OUT_SIZE);

	print_params("FC6_OUT : ", fc6_out, FC6_OUT_SIZE);
	check_params(fc6_out, "./txt/fc6_out.txt", FC6_OUT_SIZE);

	print_params("FC7_OUT : ", fc7_out, FC7_OUT_SIZE);
	check_params(fc7_out, "./txt/fc7_out.txt", FC7_OUT_SIZE);

	print_params("FC8_OUT : ", fc8_out, FC8_OUT_SIZE);
	check_params(fc8_out, "./txt/softmax_out.txt", FC8_OUT_SIZE);
  }

  return 0;

}
