#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <cstdlib>
#include <omp.h>

using namespace std;

#define SYSTEMTIME clock_t
 
void OnMult(int m_ar, int m_br) 
{
    
    char st[100];
    double temp;
    int i, j, k;

    double *pha, *phb, *phc;

    pha = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phb = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phc = (double *)malloc((m_ar * m_ar) * sizeof(double));

    for(i = 0; i < m_ar; i++)
        for(j = 0; j < m_ar; j++)
            pha[i*m_ar + j] = 1.0;

    for(i = 0; i < m_br; i++)
        for(j = 0; j < m_br; j++)
            phb[i*m_br + j] = (double)(i + 1);

    double start = omp_get_wtime();

    #pragma omp parallel
    for (int i = 0; i < m_ar; i++) {
        for (int j = 0; j < m_ar; j++) {
            double temp = 0;
            #pragma omp parallel for  // declared inside = private to each thread
            for (int k = 0; k < m_ar; k++) {
                temp += pha[i*m_ar+k] * phb[k*m_br+j];
            }
            phc[i*m_ar+j] = temp;
        }
    }
    double end = omp_get_wtime();
    cout << "Parallel Time: " << end - start << " s" << endl;


    cout << "Result matrix: " << endl;
    for(i = 0; i < 1; i++)
    {
        for(j = 0; j < min(10, m_br); j++)
            cout << phc[j] << " ";
    }
    cout << endl;

    free(pha);
    free(phb);
    free(phc);
}

// Line-by-line matrix multiplication
void OnMultLine(int m_ar, int m_br)
{
    
    char st[100];
    double temp;
    int i, j, k;

    double *pha, *phb, *phc;

    pha = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phb = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phc = (double *)malloc((m_ar * m_ar) * sizeof(double));

    for(i = 0; i < m_ar; i++)
        for(j = 0; j < m_ar; j++)
            pha[i*m_ar + j] = 1.0;

    for(i = 0; i < m_br; i++)
        for(j = 0; j < m_br; j++)
            phb[i*m_br + j] = (double)(i + 1);

    double start = omp_get_wtime();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m_ar; i++) {
        for (int k = 0; k < m_ar; k++) {
            double temp = pha[i*m_ar + k];  // private to each thread
            for (int j = 0; j < m_br; j++) {
                phc[i*m_ar + j] += temp * phb[k*m_ar + j];
            }
        }
    }
    
    double end = omp_get_wtime();
    cout << "Parallel Time: " << end - start << " s" << endl;

    cout << "Result matrix: " << endl;
    for(i = 0; i < 1; i++)
    {
        for(j = 0; j < min(10, m_br); j++)
            cout << phc[j] << " ";
    }
    cout << endl;

    free(pha);
    free(phb);
    free(phc);
}


// Block matrix multiplication
void OnMultBlock(int m_ar, int m_br, int bkSize)
{
    // assuming m_ar == m_br
    int N = m_ar;

    char st[100];

    double *pha, *phb, *phc;

    pha = (double*)malloc(N * N * sizeof(double));
    phb = (double*)malloc(N * N * sizeof(double));
    phc = (double*)malloc(N * N * sizeof(double));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            pha[i * N + j] = 1.0;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            phb[i * N + j] = (double)(i + 1);

    for (int i = 0; i < N * N; i++)
        phc[i] = 0.0;

    double start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < N; ii += bkSize) {
        int iMax = std::min(ii + bkSize, N);
        for (int kk = 0; kk < N; kk += bkSize) {
            int kMax = std::min(kk + bkSize, N);
            for (int jj = 0; jj < N; jj += bkSize) {
                int jMax = std::min(jj + bkSize, N);
                for (int i = ii; i < iMax; i++) {
                    for (int k = kk; k < kMax; k++) {
                        double a = pha[i * N + k];
                        for (int j = jj; j < jMax; j++) {
                            phc[i * N + j] += a * phb[k * N + j];
                        }
                    }
                }
            }
        }
    }

    double end = omp_get_wtime();
    cout << "Parallel Block Time: " << end - start << " s" << endl;

    cout << "Result matrix: " << endl;
    for (int j = 0; j < std::min(10, N); j++)
        cout << phc[j] << " ";
    cout << endl;

    free(pha);
    free(phb);
    free(phc);
}


int main(int argc, char *argv[])
{
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <method> <size> [blockSize]" << endl;
        cout << "  method: 1=OnMult, 2=OnMultLine, 3=OnMultBlock" << endl;
        return 1;
    }

    int op = atoi(argv[1]);
    int lin = atoi(argv[2]);
    int col = lin;
    int blockSize = (argc >= 4) ? atoi(argv[3]) : 128;

    switch (op) {
        case 1:
            OnMult(lin, col);
            break;
        case 2:
            OnMultLine(lin, col);
            break;
        case 3:
            OnMultBlock(lin, col, blockSize);
            break;
        default:
            cout << "Invalid method." << endl;
            return 1;
    }

    return 0;
}
