import cffi
import os

ffi = cffi.FFI()
defs = '''
float pearson_correlation(float *independent, float *dependent, int size); 
float var(float *data, int size);
void svd(double *h_A, double *h_U, double *h_V, double *h_S, int nrows, int ncols, int library);
void two_d_var(uint64_t data, int rows, int cols, uint64_t result);
void np_svd(uint64_t h_A, uint64_t h_U, uint64_t h_V, uint64_t h_S, int nrows, int ncols, int library);
float np_pearson_correlation(uint64_t independent, uint64_t dependent, int size);
'''
ffi.cdef(defs, override=False)
source = '''
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cusolverDn.h>
#include <cuda_runtime_api.h>
#include "magma_v2.h"
#include "magma_lapack.h"

#define min(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b; })

#include<stdlib.h>
#include<stdbool.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<stdint.h>

//--------------------------------------------------------
// FUNCTION PROTOTYPES
//--------------------------------------------------------
static float arithmetic_mean(float *data, int size);
static float mean_of_products(float *data1, float *data2, int size);
static float standard_deviation(float *data, int size);
float var(float *data, int size);
void cusolver_svd(double *h_A, double *h_U, double *h_V, double *h_S, int nrows, int ncols);
void magma_svd(double *h_A, double *h_U, double *h_V, double *h_S, int nrows, int ncols);
void svd(double *h_A, double *h_U, double *h_V, double *h_S, int nrows, int ncols, int library);
float pearson_correlation(float *independent, float *dependent, int size);

//--------------------------------------------------------
// FUNCTION pearson_correlation
//--------------------------------------------------------
float np_pearson_correlation(uint64_t independent, uint64_t dependent, int size)
{
    float *independent_pointer = (float *)independent;
    float *dependent_pointer = (float *)dependent;
    return pearson_correlation(independent_pointer, dependent_pointer, size);
}

//--------------------------------------------------------
// FUNCTION pearson_correlation
//--------------------------------------------------------
float pearson_correlation(float *independent, float *dependent, int size)
{
    float rho;

    // covariance
    float independent_mean = arithmetic_mean(independent, size);
    float dependent_mean = arithmetic_mean(dependent, size);
    float products_mean = mean_of_products(independent, dependent, size);
    float covariance = products_mean - (independent_mean * dependent_mean);

    // standard deviations of independent values
    float independent_standard_deviation = standard_deviation(independent, size);

    // standard deviations of dependent values
    float dependent_standard_deviation = standard_deviation(dependent, size);

    // Pearson Correlation Coefficient
    rho = covariance / (independent_standard_deviation * dependent_standard_deviation);

    return rho;
}

//--------------------------------------------------------
// FUNCTION arithmetic_mean
//--------------------------------------------------------
static float arithmetic_mean(float *data, int size)
{
    float total = 0;

    // note that incrementing total is done within the for loop
    for(int i = 0; i < size; total += *(data + i), i++);

    return total / size;
}


//--------------------------------------------------------
// FUNCTION two dimensional variance
//--------------------------------------------------------
void two_d_var(uint64_t data, int rows, int cols, uint64_t result)
{
    float * data_p = (float *) data;
    float * result_p = (float *) result;
    for(int k = 0; k < rows; *(result_p+k) = var((data_p+(k*cols)), cols), k++);
}


//--------------------------------------------------------
// FUNCTION variance
//--------------------------------------------------------
float var(float *data, int size)
{
    float mean = arithmetic_mean(data, size);
    float total = 0;

    // note that incrementing total is done within the for loop
    for(int k = 0; k < size; total += pow(*(data+k) - mean, 2), k++);

    return total;
}

//--------------------------------------------------------
// FUNCTION mean_of_products
//--------------------------------------------------------
static float mean_of_products(float *data1, float *data2, int size)
{
    float total = 0;

    // note that incrementing total is done within the for loop
    for(int i = 0; i < size; total += (*(data1+i) * *(data2+i)), i++);

    return total / size;
}

//--------------------------------------------------------
// FUNCTION standard_deviation
//--------------------------------------------------------
static float standard_deviation(float *data, int size)
{
    float squares[size];

    for(int i = 0; i < size; i++)
    {
        squares[i] = pow(*(data+i), 2);
    }

    float mean_of_squares = arithmetic_mean(squares, size);
    float mean = arithmetic_mean(data, size);
    float square_of_mean = pow(mean, 2);
    float variance = mean_of_squares - square_of_mean;
    float std_dev = sqrt(variance);

    return std_dev;
}


//--------------------------------------------------------
// FUNCTION numpy ctype.data svd
//--------------------------------------------------------
void np_svd(uint64_t h_A, uint64_t h_U, uint64_t h_V, uint64_t h_S, int nrows, int ncols, int library)
{
    double *h_A_pointer = (double *) h_A;
    double *h_U_pointer = (double *) h_U;
    double *h_V_pointer = (double *) h_V;
    double *h_S_pointer = (double *) h_S;
    svd(h_A_pointer, h_U_pointer, h_V_pointer, h_S_pointer, nrows, ncols, library);
}

void svd(double *h_A, double *h_U, double *h_V, double *h_S, int nrows, int ncols, int library)
{
    if (library == 0) {
        cusolver_svd(h_A, h_U, h_V, h_S, nrows, ncols);
    }
    if (library == 1) {
        magma_svd(h_A, h_U, h_V, h_S, nrows, ncols);
    }
}


void magma_svd(double *h_A, double *h_U, double *h_V, double *h_S, int nrows, int ncols)
{

  magma_init ();  //  initialize  Magma

  //  Matrix  size
  magma_int_t m=(magma_int_t)nrows , n=(magma_int_t)ncols;
  magma_int_t  info;
  double  *h_work; //  h_work  - workspace
  magma_int_t  lwork; //  workspace  size
  
  // Calculate optimal lwork
  double  *optimal_lwork;
  lwork = -1;
  magma_dmalloc_cpu(&optimal_lwork ,sizeof(double));
  magma_dgesvd(MagmaAllVec,MagmaAllVec,m,n,h_A,m,h_S,h_U,m,h_V,n,optimal_lwork,lwork,&info);
  lwork = (magma_int_t)optimal_lwork[0];

  //  Allocate  host  memory
  magma_dmalloc_pinned (&h_work ,lwork); // host  mem. for  h_work
  
  printf("%d", (int)info);printf("%c", '\\n');
  printf("%d", (int)lwork);printf("%c", '\\n');

  //  compute  the  singular  value  decomposition  of a
  magma_dgesvd(MagmaAllVec,MagmaAllVec,m,n,h_A,m,h_S,h_U,m,h_V,n,h_work,lwork,&info);

  // Free  memory
  magma_free_pinned(h_work );                  // free  host  memory
  free(optimal_lwork);
  magma_finalize( );                              //  finalize  Magma
}


//--------------------------------------------------------
// FUNCTION svd
//--------------------------------------------------------
void cusolver_svd(double *h_A, double *h_U, double *h_V, double *h_S, int nrows, int ncols)
{

	// --- gesvd only supports Nrows >= Ncols
	// --- column major memory ordering

	const int Nrows = nrows;
	const int Ncols = ncols;

	// --- cuSOLVE input/output parameters/arrays
	int work_size = 0;
	int *devInfo;
    cudaMalloc((void**)&devInfo, sizeof(int));

	// --- CUDA solver initialization
	cusolverDnHandle_t solver_handle;
	cusolverDnCreate(&solver_handle);

	// --- Setting the device matrix and moving the host matrix to the device
	double *d_A;	cudaMalloc((void**)&d_A,		Nrows * Ncols * sizeof(double));
	cudaMemcpy(d_A, h_A, Nrows * Ncols * sizeof(double), cudaMemcpyHostToDevice);

	// --- device side SVD workspace and matrices
	double *d_U;	cudaMalloc((void**)&d_U, Nrows * Nrows * sizeof(double));
	double *d_V;	cudaMalloc((void**)&d_V, Ncols * Ncols * sizeof(double));
	double *d_S;	cudaMalloc((void**)&d_S, min(Nrows, Ncols) * sizeof(double));

	// --- CUDA SVD initialization
	cusolverDnDgesvd_bufferSize(solver_handle, Nrows, Ncols, &work_size);
	double *work;	cudaMalloc((void**)&work, work_size * sizeof(double));

	// --- CUDA SVD execution
	cusolverDnDgesvd(solver_handle, 'A', 'A', Nrows, Ncols, d_A, Nrows, d_S, d_U, Nrows, d_V, Ncols, work, work_size, NULL, devInfo);
	int devInfo_h = 0;	cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

	// --- Moving the results from device to host
	cudaMemcpy(h_S, d_S, min(Nrows, Ncols) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_U, d_U, Nrows * Nrows     * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_V, d_V, Ncols * Ncols     * sizeof(double), cudaMemcpyDeviceToHost);

	cusolverDnDestroy(solver_handle);
}
'''

# Find cuda paths
CUDA_ROOT = os.environ.get('CUDA_ROOT', None) or '/usr/local/cuda'
if not os.path.isdir(CUDA_ROOT):
    raise ValueError("specified CUDA_ROOT is not a valid directory")

cuda_lib_path = os.path.join(CUDA_ROOT, 'lib64')
if not os.path.isdir(cuda_lib_path):
    cuda_lib_path = os.path.join(CUDA_ROOT, 'lib')
    if not os.path.isdir(cuda_lib_path):
        raise ValueError("cuda library path not found.  please specify "
                         "CUDA_ROOT.")

cuda_include_path = os.path.join(CUDA_ROOT, 'include')
if not os.path.isdir(cuda_include_path):
    raise ValueError("cuda include path not found.  please specify CUDA_ROOT.")

# Find magma paths
MAGMA_ROOT = os.environ.get('MAGMA_ROOT', None) or '/usr/local/magma'
if not os.path.isdir(MAGMA_ROOT):
    raise ValueError("specified MAGMA_ROOT is not a valid directory")

magma_lib_path = os.path.join(MAGMA_ROOT, 'lib64')
if not os.path.isdir(magma_lib_path):
    magma_lib_path = os.path.join(MAGMA_ROOT, 'lib')
    if not os.path.isdir(magma_lib_path):
        raise ValueError("magma library path not found.  please specify "
                         "MAGMA_ROOT.")

magma_include_path = os.path.join(MAGMA_ROOT, 'include')
if not os.path.isdir(magma_include_path):
    raise ValueError("magma include path not found.  please specify MAGMA_ROOT.")


ffi.set_source(module_name="code_analysis._cuda_functions", source=source,
               libraries=['m', 'cusolver', 'cuda', 'cudart', 'magma'],
               include_dirs=[cuda_include_path, magma_include_path],
               library_dirs=[cuda_lib_path, magma_lib_path],
               extra_compile_args=["-DADD_"])
ffi.compile(verbose=True, debug=True)
