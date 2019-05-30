#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int max(int a, int b) { return a>b?a:b; }
int min(int a, int b) { return a<b?a:b; }

void print_params(char *name, float *array, int size) {
  int i;
  printf("%s\n", name);
  for (i = 0; i < 3; i++)
	printf("%f, ",array[i]);
  printf("... ");
  for (i = 2; i >= 0; i--)
	printf(", %f",array[size-i-1]);
  printf("\n");
  fflush(stdout);
}

void print_all_params(float *array, int size) {
  int i;
  for (i = 0; i < size; i++) {
	printf("%6d : %f\n", i, array[i]);
  }
  fflush(stdout);
}

void read_params(char *path, float *array, int size) {
  int i;
  FILE *fp;
  if((fp = fopen( path , "r" )) == NULL ) {
	printf( "fileopen error\n" ) ;
	exit(1);
  }
  for (i = 0; i < size; i++)
	if (fscanf(fp, "%f\n", &array[i])!='\0');
  fclose(fp);
}

void write_params(char *path, float *array, int size) {
  int i;
  FILE *fp;
  if((fp = fopen( path , "w" )) == NULL ) {
	printf( "fileopen error\n" ) ;
	exit(1);
  }
  for (i = 0; i < size; i++)
	fprintf(fp, "%f\n", array[i]);
  fclose(fp);
}

void check_params(float *array1, char *path, int size) {
  int i, miss = 0;
  float *debug;

  debug = (float *)malloc(sizeof(float)*size);
  read_params(path, debug, size);
  //print_params("DEBUG : ", debug, size);
  for (i = 0; i < size; i++) {
	if (fabs(array1[i] - debug[i]) > 0.01) {
	  printf("%d:%f, %f\n", i, array1[i], debug[i]);
	  miss++;
	}
  }
  printf("  check params ... miss = %d\n\n", miss);
  fflush(stdout);
}

void read_binary(char *path, float *array, int size) {
  int i;
  FILE *fp;
  if((fp = fopen( path , "r" )) == NULL ) {
	printf( "fileopen error\n" ) ;
	exit(1);
  }
  fread(array, sizeof(float), size, fp);
  fclose(fp);
}

void write_binary(char *path, float *array, int size) {
  int i;
  FILE *fp;
  if((fp = fopen( path , "w" )) == NULL ) {
	printf( "fileopen error\n" ) ;
	exit(1);
  }
  fwrite(array, sizeof(float), size, fp);
  fclose(fp);
}

void check_binary(float *array1, char *path, int size) {
  int i, miss = 0;
  float *debug;

  debug = (float *)malloc(sizeof(float)*size);
  read_binary(path, debug, size);
  //print_params("DEBUG : ", debug, size);
  for (i = 0; i < size; i++) {
	//printf("%d:%f, %f\n", i, array1[i], array2[i]);
	if (fabs(array1[i] - debug[i]) > 0.01) {
	  miss++;
	}
  }
  printf("  check binary ... miss = %d\n\n", miss);
  fflush(stdout);
}

void padding(float *input, int isize, int ichan, float *output, int pad) {
  int ocol, orow, och;
  int osize=isize+pad+pad;
  for (och = 0; och < ichan; och++) {
	for (orow = 0; orow < osize; orow++) {
	  for (ocol = 0; ocol < osize; ocol++) {
		*(output+och*osize*osize+orow*osize+ocol) = (float)0.0;
	  }
	}
  }
  
  for (och = 0; och < ichan; och++) {
	for (orow = 0; orow < isize; orow++) {
	  for (ocol = 0; ocol < isize; ocol++) {
		*(output+och*osize*osize+(orow+pad)*osize+(ocol+pad)) = *(input+och*isize*isize+orow*isize+ocol);
	  }
	}
  }
  
}

void convolution(float *input, int isize, int ichan, float *output, int osize, int ochan, float *weight, float *bias, int ksize, int stride){
  /*
	Data Format:
	input[ch (< ichan)][row (< isize)][[col (< isize)]
	output[ch (< ochan)][row (< osize)][col (< osize)]
	weight[karnel (< ochan)][ch (< ichan)][row (< ksize)][col (< ksize)]
	bias[karnel (< ochan)]
   */
  int ocol, orow, och, kcol, krow, kch;

  printf("Convolution:\n");
  printf("  isize=%d, ichan=%d, osize=%d, ochan=%d, ksize=%d, stride=%d\n", isize, ichan, osize, ochan, ksize, stride);
  for (och= 0; och < ochan; och++) {
	for (orow = 0; orow < osize; orow++) {
	  for (ocol = 0; ocol < osize; ocol++) {
		*(output+och*osize*osize+orow*osize+ocol) = 0.0;
		for (krow = 0; krow < ksize; krow++) {
		  for (kcol = 0; kcol < ksize; kcol++) {
			for (kch = 0; kch < ichan; kch++) {
			  // output[och][ocol][orow] += weight[och][kch][kcol][krow] * input[kch][kcol + ocol*stride][krow + orow*stride];
			  // example : conv1_out[57] += conv1_w[i*11*11+j*11+k] * image[(227*4*1+4*2)+i*227*227+j*227+k];
			  *(output+och*osize*osize+orow*osize+ocol) += *(weight+och*ichan*ksize*ksize+kch*ksize*ksize+krow*ksize+kcol) *
				*(input+kch*isize*isize+krow*isize+kcol+(orow*isize*stride+ocol*stride));
			}
		  }
		}
		*(output+och*osize*osize+orow*osize+ocol) += *(bias+och);
	  }
	}
  }
  printf("\n");fflush(stdout);
}

void maxpooling(float *input, int isize, int ichan, float *output, int osize,  int ksize, int stride) {
  int ocol, orow, och, kcol, krow;
  float max, tmp;

  printf("MaxPooling:\n");
  printf("  isize=%d, ichan=%d, osize=%d, ksize=%d, stride=%d\n", isize, ichan, osize, ksize, stride);

  for (och= 0; och < ichan; och++) {
	for (orow = 0; orow < osize; orow++) {
	  for (ocol = 0; ocol < osize; ocol++) { 
		max = -256.0;
		for (krow = 0; krow < ksize; krow++) {
		  for (kcol = 0; kcol < ksize; kcol++) {
			tmp = *(input+och*isize*isize+krow*isize+kcol+(orow*isize*stride+ocol*stride));
			//tmp=input[och][orow+krow][kcol*ocol]
      if (max < tmp) max = tmp;
		  }
		*(output+och*osize*osize+osize*orow+ocol) = max;
		}
	  }
	}
  }
  printf("\n");fflush(stdout);
}

void relu(float *input, int isize, int ichan) {
  int ocol, orow, och;

  printf("ReLu:\n");
  printf("  isize=%d, ichan=%d\n", isize, ichan);

  for (och= 0; och < ichan; och++) {
	for (orow = 0; orow < isize; orow++) {
	  for (ocol = 0; ocol < isize; ocol++) {
		if (*(input+och*isize*isize+orow*isize+ocol) < 0.0) *(input+och*isize*isize+orow*isize+ocol) = 0.0;
	  }
	}
  }
  printf("\n");fflush(stdout);
}

void lrn(float *input, int isize, int ichan, float *output, int k, int n, float alpha, float beta) {
  int ocol, orow, och, i, j;
  float sum, tmp;

  alpha = 0.0001;beta = 0.75;
  printf("LRN:\n");
  printf("  isize=%d, ichan=%d, k=%d, n=%d, a=%f, b=%f\n", isize, ichan, k, n, alpha, beta);

  for (och= 0; och < ichan; och++) {
	for (orow = 0; orow < isize; orow++) {
	  for (ocol = 0; ocol < isize; ocol++) {
		sum = 0.0;
		for (j = max(0, och-(n/2)); j <= min(ichan-1, och+(n/2)); j++) {
		  tmp = *(input+j*isize*isize+orow*isize+ocol);
		  sum += tmp * tmp;
		}
		*(output+och*isize*isize+orow*isize+ocol) = 
		  *(input+och*isize*isize+orow*isize+ocol) *
		  powf((float)((float)k + alpha / (float) n * sum), (float)-beta);
	  }
	}
  }
  printf("\n");fflush(stdout);
}


void classifier(float *input, int isize, float *output, int osize, float *weight, float *bias) {
  int i, j;

  printf("Classifier:\n");
  printf("  isize=%d, osize=%d\n", isize, osize);
  
  for (i = 0; i < osize; i++) {
	*(output+i) = 0.0;
	
	for(j = 0; j < isize; j++) {
	  if (i == 0 && j < 20 && isize == 4096 && osize == 4096) printf("%f x %f\n", *(weight+i*isize+j),  *(input+j));
	  if (i == 0 && j < 20 && isize == 4096 && osize == 4096) printf("%X x %X\n", (int16_t)(*(weight+i*isize+j) * 4096) ,  (int16_t)(*(input+j) * 4096));
	  *(output+i) += *(weight+i*isize+j) * *(input+j);
	  if (i == 0 && j < 20 && isize == 4096 && osize == 4096) printf("sum %f\n", *(output+i));
	  if (i == 0 && j < 20 && isize == 4096 && osize == 4096) printf("sum  %X\n", (int16_t)(*(output+i) * 4096));
	  
	}
	if (i == 0 && isize == 4096 && osize == 4096) printf("%f, %X\n", *(output+i), (int16_t)(*(output+i) * 4096));
	*(output+i) += *(bias+i);
	if (i == 0 && isize == 4096 && osize == 4096) printf("%f, %X\n", *(output+i), (int16_t)(*(output+i) * 4096));
  }
  printf("\n");fflush(stdout);
}

void softmax(float *input, int isize) {
  int i, j;
  float sum = 0.0;

  printf("Softmax:\n");
  printf("  isize=%d\n", isize);
  
  for (i = 0; i < isize; i++) {
	sum += expf(*(input + i));
  }
  for (i = 0; i < isize; i++) {
	*(input+i) = expf(*(input + i)) / sum;
  }
  printf("\n");
  fflush(stdout);
}

void show_result(float *softmax, char *path, int size) {
  int first = 0, second = 0, third = 0;
  int i;
  FILE *fp;
  char category[size][64];
  char tmp[64];

  if((fp = fopen( path , "r" )) == NULL ) {
	printf( "fileopen error\n" ) ;
	exit(1);
  }
  for (i = 0; i < size; i++)
	if (fscanf(fp, "%s %[^\n]\n", tmp, category[i])!='\0');
  fclose(fp);
  
  printf("Show result: \n");
  
  for (i = 0; i < size; i++){
	if (softmax[i] > softmax[third]) {
	  third = i;
	  if (softmax[i] > softmax[second]) {
		third = second;
		second = i;
		if (softmax[i] > softmax[first]) {
		  second = first;
		  first = i;
		}
	  }
	}
  }
  printf("  %s : %f\n", category[first], softmax[first]*100);
  printf("  %s : %f\n", category[second], softmax[second]*100);
  printf("  %s : %f\n", category[third], softmax[third]*100);
  printf("\n");
  fflush(stdout);
}
