#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <stdbool.h>

#ifdef _MSC_VER
#include <malloc.h>
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#define ALIGN 32 
#define MAX_DIM 32

typedef struct {
    float mape;
    int mask;
} ResultItem;

static inline void max_heapify(ResultItem* heap, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && heap[left].mape > heap[largest].mape)
        largest = left;

    if (right < n && heap[right].mape > heap[largest].mape)
        largest = right;

    if (largest != i) {
        ResultItem temp = heap[i];
        heap[i] = heap[largest];
        heap[largest] = temp;
        max_heapify(heap, n, largest);
    }
}

static inline void heap_insert_or_replace(ResultItem* heap, int* current_size, int capacity, float mape, int mask) {
    if (*current_size < capacity) {
        int i = *current_size;
        heap[i].mape = mape;
        heap[i].mask = mask;
        (*current_size)++;
        
        if (*current_size == capacity) {
            for (int j = capacity / 2 - 1; j >= 0; j--)
                max_heapify(heap, capacity, j);
        }
    } else {
        if (mape < heap[0].mape) {
            heap[0].mape = mape;
            heap[0].mask = mask;
            max_heapify(heap, capacity, 0);
        }
    }
}

static int compare_results(const void* a, const void* b) {
    float diff = ((ResultItem*)a)->mape - ((ResultItem*)b)->mape;
    return (diff > 0) - (diff < 0);
}

static inline float hsum256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ps(sums, shuf);
    return _mm_cvtss_f32(sums);
}

void transpose_pad_matrix(const float* src, float* dst, int rows, int cols, int padded_rows) {
    memset(dst, 0, cols * padded_rows * sizeof(float));
    for (int c = 0; c < cols; c++) {
        for (int r = 0; r < rows; r++) {
            dst[c * padded_rows + r] = src[r * cols + c];
        }
    }
}

typedef struct {
    float* XtX_fold;     
    float* Xty_fold;     
    float* X_test_T;     
    float* y_test;       
    int test_cnt;        
    int test_cnt_pad;    
    int m;
} CVFold;

static inline int solve_cholesky(int n, float* A, float* b, float* x) {
    float L[MAX_DIM * MAX_DIM];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = A[i * n + j];
            for (int k = 0; k < j; k++) sum -= L[i * n + k] * L[j * n + k];
            if (i == j) {
                if (sum <= 1e-12f) return -1;
                L[i * n + j] = sqrtf(sum);
            } else L[i * n + j] = sum / L[j * n + j];
        }
    }
    float y[MAX_DIM];
    for (int i = 0; i < n; i++) {
        float sum = b[i];
        for (int j = 0; j < i; j++) sum -= L[i * n + j] * y[j];
        y[i] = sum / L[i * n + i];
    }
    for (int i = n - 1; i >= 0; i--) {
        float sum = y[i];
        for (int j = i + 1; j < n; j++) sum -= L[j * n + i] * x[j];
        x[i] = sum / L[i * n + i];
    }
    return 0;
}

static inline float compute_fold_mape(const CVFold* fold, const int* idxs, const float* beta, int k) {
    int n_pad = fold->test_cnt_pad;
    int actual_n = fold->test_cnt;
    const float* X_T = fold->X_test_T;
    const float* y_true = fold->y_test;

    __m256 sum_ratio = _mm256_setzero_ps();
    __m256 v_eps = _mm256_set1_ps(1e-8f);
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    
    for (int i = 0; i < n_pad; i += 8) {
        __m256 v_pred = _mm256_setzero_ps();
        for (int j = 0; j < k; j++) {
            __m256 v_beta = _mm256_set1_ps(beta[j]);
            __m256 v_x = _mm256_load_ps(&X_T[idxs[j] * n_pad + i]);
            v_pred = _mm256_fmadd_ps(v_beta, v_x, v_pred);
        }
        __m256 v_y = _mm256_load_ps(&y_true[i]);
        __m256 diff = _mm256_sub_ps(v_y, v_pred);
        __m256 v_diff = _mm256_and_ps(diff, sign_mask);
        __m256 v_abs_y = _mm256_and_ps(v_y, sign_mask);
        __m256 v_denom = _mm256_max_ps(v_abs_y, v_eps);
        __m256 v_ratio = _mm256_div_ps(v_diff, v_denom);
        sum_ratio = _mm256_add_ps(sum_ratio, v_ratio);
    }
    return hsum256_ps(sum_ratio) / actual_n * 100.0f;
}

EXPORT int exhaustive_feature_selection(
    float* X, float* y, int n, int m, int k_max, 
    int k_folds, int* fold_indices, 
    float* out_mapes, int* out_combinations, int* out_lens,
    int top_k 
) {
    // 1. Global Precomputation
    float* XtX_G = _mm_malloc(m * m * sizeof(float), ALIGN);
    float* Xty_G = _mm_malloc(m * sizeof(float), ALIGN);
    memset(XtX_G, 0, m * m * sizeof(float));
    memset(Xty_G, 0, m * sizeof(float));

    for (int i = 0; i < n; i++) {
        float yi = y[i];
        const float* Xi = &X[i * m];
        for (int r = 0; r < m; r++) {
            float val_r = Xi[r];
            Xty_G[r] += val_r * yi;
            for (int c = 0; c <= r; c++) XtX_G[r * m + c] += val_r * Xi[c];
        }
    }
    for(int r=0; r<m; r++) for(int c=0; c<r; c++) XtX_G[c*m+r] = XtX_G[r*m+c];

    // 2. Prepare Folds
    CVFold* folds = malloc(k_folds * sizeof(CVFold));
    int* fold_counts = calloc(k_folds, sizeof(int));
    for(int i=0; i<n; i++) fold_counts[fold_indices[i]]++;

    for (int f = 0; f < k_folds; f++) {
        int cnt = fold_counts[f];
        int cnt_pad = (cnt + 7) & ~7; 
        folds[f].m = m;
        folds[f].test_cnt = cnt;
        folds[f].test_cnt_pad = cnt_pad;
        folds[f].XtX_fold = _mm_malloc(m * m * sizeof(float), ALIGN);
        folds[f].Xty_fold = _mm_malloc(m * sizeof(float), ALIGN);
        folds[f].X_test_T = _mm_malloc(m * cnt_pad * sizeof(float), ALIGN);
        folds[f].y_test   = _mm_malloc(cnt_pad * sizeof(float), ALIGN);

        memset(folds[f].XtX_fold, 0, m * m * sizeof(float));
        memset(folds[f].Xty_fold, 0, m * sizeof(float));
        float* temp_X_rows = malloc(cnt * m * sizeof(float));
        int idx = 0;
        for (int i = 0; i < n; i++) {
            if (fold_indices[i] == f) {
                float yi = y[i];
                folds[f].y_test[idx] = yi;
                const float* Xi = &X[i * m];
                memcpy(&temp_X_rows[idx * m], Xi, m * sizeof(float));
                for (int r = 0; r < m; r++) {
                    float val_r = Xi[r];
                    folds[f].Xty_fold[r] += val_r * yi;
                    for (int c = 0; c < m; c++) folds[f].XtX_fold[r*m + c] += val_r * Xi[c];
                }
                idx++;
            }
        }
        transpose_pad_matrix(temp_X_rows, folds[f].X_test_T, idx, m, cnt_pad);
        free(temp_X_rows);
    }
    free(fold_counts);

    // 3. Task Generation
    int loop_limit = 1 << (m - 1);
    int* valid_masks = malloc(loop_limit * sizeof(int));
    int total_tasks = 0;
    for (int mask = 0; mask < loop_limit; mask++) {
        int bits = 0; 
        int t = mask;
        while(t) { bits += t & 1; t >>= 1; }
        if (bits < k_max) valid_masks[total_tasks++] = mask;
    }

    bool use_top_k = (top_k > 0);
    int num_threads = omp_get_max_threads();
    ResultItem** thread_heaps = NULL;
    int* heap_counts = NULL;

    if (use_top_k) {
        thread_heaps = malloc(num_threads * sizeof(ResultItem*));
        heap_counts = calloc(num_threads, sizeof(int));
        for(int i=0; i<num_threads; i++) {
            thread_heaps[i] = malloc(top_k * sizeof(ResultItem));
        }
    }

    #pragma omp parallel
    {
        float A_local[MAX_DIM * MAX_DIM];
        float b_local[MAX_DIM];
        float x_local[MAX_DIM];
        int idxs[MAX_DIM];

        int tid = omp_get_thread_num();
        ResultItem* my_heap = use_top_k ? thread_heaps[tid] : NULL;
        int my_count = 0;

        int t;
        #pragma omp for schedule(dynamic, 64)
        for (t = 0; t < total_tasks; t++) {
            int mask = valid_masks[t];
            int k = 0;
            for (int bit = 0; bit < m - 1; bit++) {
                if ((mask >> bit) & 1) idxs[k++] = bit;
            }
            idxs[k++] = m - 1;

            float fold_mapes = 0;
            const float ridge = 1e-4f; 
            bool singular = false;

            for (int f = 0; f < k_folds; f++) {
                for (int r = 0; r < k; r++) {
                    int gr = idxs[r];
                    b_local[r] = Xty_G[gr] - folds[f].Xty_fold[gr];
                    for (int c = 0; c <= r; c++) {
                        int gc = idxs[c];
                        float val = XtX_G[gr * m + gc] - folds[f].XtX_fold[gr * m + gc];
                        if (r == c) val += ridge;
                        A_local[r * k + c] = val;
                    }
                }
                if (solve_cholesky(k, A_local, b_local, x_local) == 0) {
                    fold_mapes += compute_fold_mape(&folds[f], idxs, x_local, k);
                } else {
                    singular = true;
                    break;
                }
            }

            float final_mape = singular ? 1e9f : (fold_mapes / k_folds);

            if (use_top_k) {
                heap_insert_or_replace(my_heap, &my_count, top_k, final_mape, mask);
            } else {
                out_mapes[t] = final_mape;
                out_lens[t] = k;
                int* row_ptr = out_combinations + (size_t)t * (m-1);
                for(int z=0; z<k; z++) row_ptr[z] = idxs[z];
            }
        }
        
        if (use_top_k) {
            heap_counts[tid] = my_count;
        }
    }

    int return_count = total_tasks;

    if (use_top_k) {
        int total_candidates = 0;
        for(int i=0; i<num_threads; i++) total_candidates += heap_counts[i];
        
        ResultItem* all_results = malloc(total_candidates * sizeof(ResultItem));
        int ptr = 0;
        for(int i=0; i<num_threads; i++) {
            memcpy(all_results + ptr, thread_heaps[i], heap_counts[i] * sizeof(ResultItem));
            ptr += heap_counts[i];
            free(thread_heaps[i]);
        }
        free(thread_heaps);
        free(heap_counts);

        qsort(all_results, total_candidates, sizeof(ResultItem), compare_results);

        return_count = (total_candidates < top_k) ? total_candidates : top_k;
        
        for(int i=0; i<return_count; i++) {
            out_mapes[i] = all_results[i].mape;
            
            int mask = all_results[i].mask;
            int k = 0;
            int idxs[MAX_DIM];
            for (int bit = 0; bit < m - 1; bit++) {
                if ((mask >> bit) & 1) idxs[k++] = bit;
            }
            idxs[k++] = m - 1;
            
            out_lens[i] = k;
            int* row_ptr = out_combinations + (size_t)i * (m-1);
            for(int z=0; z<k; z++) row_ptr[z] = idxs[z];
        }
        free(all_results);
    }

    _mm_free(XtX_G);
    _mm_free(Xty_G);
    for(int f=0; f<k_folds; f++) {
        _mm_free(folds[f].XtX_fold);
        _mm_free(folds[f].Xty_fold);
        _mm_free(folds[f].X_test_T);
        _mm_free(folds[f].y_test);
    }
    free(folds);
    free(valid_masks);
    
    return return_count;
}