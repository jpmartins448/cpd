#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <cstdlib>

using namespace std;

#define SYSTEMTIME clock_t

 
void OnMult(int m_ar, int m_br) 
{
    SYSTEMTIME Time1, Time2;
    
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

    Time1 = clock();

    for(i = 0; i < m_ar; i++)
    {
        for(j = 0; j < m_br; j++)
        {
            temp = 0;
            for(k = 0; k < m_ar; k++)
            {    
                temp += pha[i*m_ar + k] * phb[k*m_br + j];
            }
            phc[i*m_ar + j] = temp;
        }
    }

    Time2 = clock();
    snprintf(st, sizeof(st), "Time: %3.3f seconds\n",
         (double)(Time2 - Time1) / CLOCKS_PER_SEC);
    cout << st;

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
double OnMultLine(int m_ar, int m_br)
{
    SYSTEMTIME Time1, Time2;
    
    char st[100];
    double temp;
    int i, j, k;

    double *pha, *phb, *phc;

    pha = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phb = (double *)malloc((m_ar * m_ar) * sizeof(double));
    phc = (double *)malloc((m_ar * m_ar) * sizeof(double));

    for(i = 0; i < m_ar; i++){
        for(j = 0; j < m_ar; j++){
            pha[i*m_ar + j] = 1.0;
            phc[i*m_ar + j] = 0.0;
        }
    }

    for(i = 0; i < m_br; i++){
        for(j = 0; j < m_br; j++){
            phb[i*m_br + j] = (double)(i + 1);
        }

    }

    
    Time1 = clock();

    for(i = 0; i < m_ar; i++)
    {
        for(j = 0; j < m_ar; j++)
        {
            for (k=0; k < m_ar; k++){
                phc[i*m_ar + k]+= pha[i*m_ar + j]*phb[j*m_ar + k];
            }
        }
    }

    Time2 = clock();
    double elapsed = (double)(Time2 - Time1) / CLOCKS_PER_SEC;
    snprintf(st, sizeof(st), "Time: %3.3f seconds\n", elapsed);
    cout << st;

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
    return elapsed;
   
}


// Block matrix multiplication
void OnMultBlock(int m_ar, int m_br, int bkSize)
{
    // assuming m_ar == m_br
    int N = m_ar;

    clock_t Time1, Time2;
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

    Time1 = clock();

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

    Time2 = clock();

    snprintf(st, sizeof(st), "Time: %3.3f seconds\n",
             (double)(Time2 - Time1) / CLOCKS_PER_SEC);
    cout << st;

    cout << "Result matrix: " << endl;
    for (int j = 0; j < std::min(10, N); j++)
        cout << phc[j] << " ";
    cout << endl;

    free(pha);
    free(phb);
    free(phc);
}


int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <matrix_size> " << endl;
        return 1;
    }
    int size   = atoi(argv[1]);
    int bkSize = atoi(argv[2]);

    cout << "Running OnMultBlock with size " << size << "x" << size
        << endl;
    OnMultLine(size, size);
    return 0;
}
