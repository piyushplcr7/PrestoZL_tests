#ifndef CUFFTWISDOMDEFSH
#define CUFFTWISDOMDEFSH

extern int cufftwisdomarraysize;

typedef struct {
  int fftlen;
  int batch_size;
} cufftwisdompair;

extern cufftwisdompair cufftwisdomarray[];

#endif